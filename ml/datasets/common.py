import enum

CLASS_INDICES = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'scc': 6,
    'vasc': 7,
    'ack': 8
}


class ClassCategory(enum.Enum):
    BENIGN = 'BENIGN'
    NOT_BENIGN = 'NOT_BENIGN'


CLASS_CATEGORIES = {
    'akiec': ClassCategory.BENIGN,
    'bcc': ClassCategory.BENIGN,
    'bkl': ClassCategory.BENIGN,
    'df': ClassCategory.BENIGN,
    'mel': ClassCategory.NOT_BENIGN,
    'nv': ClassCategory.BENIGN,
    'scc': ClassCategory.BENIGN,
    'vasc': ClassCategory.BENIGN,
    'ack': ClassCategory.NOT_BENIGN
}

CLASS_DESCRIPTIONS = {
    'akiec': '\nActinic keratoses - '
             '\nTypically, less than 2 cm, or about the size of a pencil eraser.'
             '\nThick, scaly, or crusty skin patch.'
             '\nAppears on parts of the body that receive a lot of sun exposure (hands, arms, face, scalp, and neck).'
             '\nUsually pink in color but can have a brown, tan, or gray base.',

    'bcc': '\nBasal Cell Carcinoma -'
           '\nRaised, firm, and pale areas that may resemble a scar.'
           '\nRaised, firm, and pale areas that may resemble a scar.'
           '\nDome-like, pink or red, shiny, and pearly areas that may have a sunk-in center, like a crater.'
           '\nVisible blood vessels on the growth.'
           '\nEasy bleeding or oozing wound that doesn’t seem to heal or heals and then reappears.',

    'bkl': '\nBenign Keratosis-Like Lesions - '
           '\nA seborrheic keratosis is a common noncancerous skin growth.'
           '\nPeople tend to get more of them as they get older.'
           '\nSeborrheic keratoses are usually brown, black or light tan.'
           '\nThe growths look waxy, scaly and slightly raised. They usually appear on the head, neck, chest or back.',

    'df': '\nDermatofibroma - '
          '\nA dermatofibroma can occur anywhere on the skin.'
          '\nDermatofibroma size varies from 0.5–1.5 cm diameter; most lesions are 7–10 mm diameter.'
          '\nA dermatofibroma is tethered to the skin surface and mobile over subcutaneous tissue.'
          '\nColour may be pink to light brown in white skin, and dark brown to black in dark skin.'
          '\nDermatofibromas do not usually cause symptoms, but they are sometimes painful, tender, or itchy.',

    'mel': '\nMelanoma - '
           '\nThe most serious form of skin cancer, more common in fair-skinned people. '
           '\nMole anywhere on the body that has irregularly shaped edges, asymmetrical shape, and multiple colors. '
           '\nMole that has changed color or gotten bigger over time.'
           '\nUsually larger than a pencil eraser.',

    'nv': '\nMelanocytic nevi - '
          '\nThe majority of moles appear during the first two decades of a persons life.'
          '\nOne in every 100 babies being born with moles.'
          '\nA mole can be either subdermal (under the skin) or a pigmented growth on the skin,'
          '\nThe high concentration of the bodys pigmenting agent, melanin, is responsible for their dark color.',

    'scc': '\nSquamous Cell Carcinoma - '
           '\nOften occurs in areas exposed to UV radiation, such as the face, ears, and back of the hands.'
           '\nScaly, reddish patch of skin progresses to a raised bump that continues to grow.'
           '\nGrowth that bleeds easily and doesn’t heal or heals and then reappears.',

    'vasc': '\nVascular Lesions - '
            '\nVascular lesions are relatively common abnormalities of the skin and underlying tissues, '
            'more commonly known as birthmarks.'
            '\nWhile these birthmarks can look similar at times, '
            'they each vary in terms of origin and necessary treatment.',

    'ack': 'Acne - '
           '\nis a skin condition that occurs when your hair follicles become plugged with oil and dead skin cells.'
           '\nIt causes whiteheads, blackheads or pimples. '
           '\nAcne is most common among teenagers, though it affects people of all ages.'

}
