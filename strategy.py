#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 12h EMA34 trend filter and volume spike confirmation.
# Long when: price breaks above 6h Camarilla R4 level AND close > 12h EMA34 AND volume > 2.0x 24-bar average
# Short when: price breaks below 6h Camarilla S4 level AND close < 12h EMA34 AND volume > 2.0x 24-bar average
# Exit via ATR(24) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 6h Camarilla for structure (proven edge from top performers), 12h EMA34 for HTF trend alignment, volume spike for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Camarilla_R4_S4_12hEMA34_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Camarilla pivots (based on previous 6h bar)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6