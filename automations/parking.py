import config
import logging
from playwright.async_api import async_playwright

# import asyncio
import os

config = config.Config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def navigate_website(ticket_number, ticket_date, ticket_time):
    try:
        if os.path.exists("screenshot.png"):
            os.remove("screenshot.png")
            logger.debug("Deleted existing screenshot.png")
    except Exception as e:
        logger.error(f"Error deleting screenshot.png: {str(e)}")
        raise e

    url = config.ocr_url
    logger.info(f"Starting website navigation for ticket {ticket_number}")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        logger.debug("Browser launched")

        # Configure browser context with optimized settings
        context = await browser.new_context(
            ignore_https_errors=True, java_script_enabled=True, bypass_csp=True
        )

        page = await context.new_page()

        await page.goto(url, wait_until="networkidle")
        logger.info("Navigated to login page")

        try:
            await page.fill("#ContentPlaceHolder1_txtUser", config.ocr_username)
            await page.fill("#ContentPlaceHolder1_txtPassword", config.ocr_password)
            await page.click("#ContentPlaceHolder1_btnLogIn")
            logger.info("Login credentials submitted")
        except Exception as e:
            logger.error(f"Error submitting login credentials: {str(e)}")
            raise e

        try:
            await page.wait_for_load_state("networkidle")
            await page.click("#mnLinks_lnkValidate")
            logger.debug("Navigated to validation page")
        except Exception as e:
            logger.error(f"Error navigating to validation page: {str(e)}")
            raise e

        try:
            await page.fill("#ContentPlaceHolder1_txtTicketNumber", ticket_number)
            await page.fill("#ContentPlaceHolder1_txtEntryDate", ticket_date)
            await page.fill("#ContentPlaceHolder1_txtEntryTime", ticket_time)
            logger.info(
                f"Filled ticket details - Date: {ticket_date}, Time: {ticket_time}"
            )
        except Exception as e:
            logger.error(f"Error filling ticket details: {str(e)}")
            raise e

        try:
            await page.click("#ContentPlaceHolder1_btnFind")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_selector("#ContentPlaceHolder1_btnAdd")
            await page.click("#ContentPlaceHolder1_btnAdd")
            logger.info("Ticket validation completed")

        except Exception as e:
            logger.error(f"Error validating ticket: {str(e)}")
            raise e

        await page.screenshot(path="screenshot.png")
        logger.debug("Screenshot captured")

        await browser.close()
        logger.info("Browser session ended")


# # # Example of how to call the function
# asyncio.run(
#     navigate_website(
#         "30612",
#         "11/06/2024",
#         "0912",
#     )
# )

# # # Example of how to call the function
# asyncio.run(
#     navigate_website(
#         "30611",
#         "11/32/2024",
#         "0912",
#     )
# )
