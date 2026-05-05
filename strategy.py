#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation
# Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close)
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND volume > 1.5x 20-period average AND close > 12h EMA50
# Short when Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND volume > 1.5x 20-period average AND close < 12h EMA50
# Exit when momentum weakens: Bull Power < 0 for long OR Bear Power > 0 for short
# Uses Elder Ray to detect momentum shifts, effective in both bull (continuation) and bear (mean reversion via exits) markets.
# Timeframe: 6h, HTF: 12h. Target: 80-180 total trades over 4 years (20-45/year).

name = "6h_ElderRay_BullBearPower_12hEMA50_Volume"
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
    
    # Calculate volume confirmation on 6h (no HTF needed)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray components on 6h
    # Bull Power = High - EMA13(close)
    # Bear Power = Low - EMA13(close)
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
        bull_power = high - ema_13
        bear_power = low - ema_13
    else:
        bull_power = np.zeros(n)
        bear_power = np.zeros(n)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND volume filter AND above 12h EMA50
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                volume_filter[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND volume filter AND below 12h EMA50
            elif (bear_power[i] < 0 and 
                  bull_power[i] < 0 and 
                  volume_filter[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative (momentum weakening)
            if bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive (momentum weakening)
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals