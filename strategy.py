#!/usr/bin/env python3
"""
1d_1day_Williams_Alligator_ElderRay_Top
Hypothesis: Williams Alligator identifies trend direction; Elder Ray confirms strength; combined with weekly trend filter to avoid counter-trend trades. Works in bull/bear by only trading with the weekly trend. Target: 15-25 trades/year (60-100 total).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: SMAs with future shift (as per original)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13.values
    bear_power = low - ema13.values
    
    # Weekly trend filter: price above/below weekly EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Trend condition: Alligator aligned (Lips > Teeth > Jaw for up, reverse for down)
    bull_alligator = (lips > teeth) & (teeth > jaw)
    bear_alligator = (lips < teeth) & (teeth < jaw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # max shift + periods
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        weekly_trend_up = close[i] > ema_1w_aligned[i]
        weekly_trend_down = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            # Long: bullish Alligator + positive Bull Power + weekly uptrend
            if bull_alligator[i] and (bull_power[i] > 0) and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator + negative Bear Power + weekly downtrend
            elif bear_alligator[i] and (bear_power[i] < 0) and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bear Power turns negative
            if bear_alligator[i] or (bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bull Power turns positive
            if bull_alligator[i] or (bull_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1day_Williams_Alligator_ElderRay_Top"
timeframe = "1d"
leverage = 1.0