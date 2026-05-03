#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1w EMA50 trend filter and 1d volume spike confirmation.
# Long when: 6h Williams %R crosses above -80 (oversold) AND close > 1w EMA50 AND volume > 2.0x 24-bar average
# Short when: 6h Williams %R crosses below -20 (overbought) AND close < 1w EMA50 AND volume > 2.0x 24-bar average
# Exit via ATR(24) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 6h Williams %R for mean reversion in extremes, 1w EMA50 for HTF trend alignment, volume spike for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_WilliamsR_1wEMA50_VolumeSpike_ATRStop_v1"
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
    
    # Calculate 6h Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1w EMA50 (HTF trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate