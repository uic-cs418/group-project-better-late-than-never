from flask import Flask, render_template, request, flash
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired
import form_selections as fields

app = Flask(__name__)
app.secret_key = "encourageexecutestrongmedicine69420!"
csrf = CSRFProtect(app)


class InputForm(FlaskForm):
    """
    Input neighborhood, category, price for pandas query
    """

    neighborhood = SelectField(
        "Neighborhood", choices=fields.chi_communities, validators=[DataRequired()]
    )
    category = SelectField(
        "Category", choices=fields.categories, validators=[DataRequired()]
    )
    price = SelectField("Price", choices=fields.prices, validators=[DataRequired()])


@app.route("/")
def index():
    form = InputForm()

    if request.method == "POST":
        if form.validate_on_submit():
            name = form.name.data
            flash(f"inputs are ", "success")
            return render_template("index.html", form=form)
        else:
            flash("CSRF Token Missing or Invalid!", "danger")

    return render_template("index.html", form=form)


if __name__ == "__main__":
    app.run()
