from flask import Flask, render_template, request, flash
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired
import pandas as pd
import form_selections as fields
import recommend as rec

# Load app and dataframe
app = Flask(__name__)
app.secret_key = "encourageexecutestrongmedicine69420!"
csrf = CSRFProtect(app)
yelp_df = pd.read_json("static/chicago_restaurants.json", lines=True)


class InputForm(FlaskForm):
    """
    Form class to input neighborhood, category, price for pandas query
    """

    neighborhood = SelectField(
        "Neighborhood", choices=fields.chi_communities, validators=[DataRequired()]
    )
    category = SelectField(
        "Category", choices=fields.categories, validators=[DataRequired()]
    )
    price = SelectField("Price", choices=fields.prices, validators=[DataRequired()])
    submit = SubmitField("Submit")


@app.route("/", methods=["GET", "POST"])
def index():
    form = InputForm()

    if request.method == "POST":
        if form.validate_on_submit():
            # Get form data
            neighborhood = form.neighborhood.data
            category = form.category.data
            price = form.price.data
            # Run query from data
            yelp_recs = rec.get_best_restaurants(yelp_df, category, neighborhood, price)
            # Return results with HTML
            for recommendation in yelp_recs:
                flash(f"{recommendation}", "success")
            return render_template("index.html", form=form)
        else:
            flash("CSRF Token Missing or Invalid!", "danger")

    return render_template("index.html", form=form)


if __name__ == "__main__":
    app.run()
