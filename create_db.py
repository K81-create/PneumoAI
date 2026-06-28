from app import app, db

with app.app_context():
    print("========== CREATING DATABASE TABLES ==========")
    db.create_all()
    print("========== DATABASE READY ==========")
