#!/usr/bin/env python3
name = "12h_1d_Camarilla_R3_S3_Breakout_Trend_Filter_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Close for calculations
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # Using previous day's values (shifted by 1)
    if len(close_1d) > 1:
        prev_close = np.roll(close_1d, 1)
        prev_high = np.roll(high_1d, 1)
        prev_low = np.roll(low_1d, 1)
        prev_close[0] = close_1d[0]  # First value
        prev_high[0] = high_1d[0]
        prev_low[0] = low_1d[0]
    else:
        prev_close = close_1d
        prev_high = high_1d
        prev_low = low_1d
    
    # Camarilla R3 and S3 levels
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 12h volume spike: > 1.8x 6-period average (3 days)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    vol_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(6, 34)  # Wait for volume MA and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above EMA34 AND breaks above R3 with volume spike
            if close[i] > ema34_1d_aligned[i] and close[i] > R3_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below EMA34 AND breaks below S3 with volume spike
            elif close[i] < ema34_1d_aligned[i] and close[i] < S3_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below EMA34 OR below S3 (reversal signal)
            if close[i] < ema34_1d_aligned[i] or close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above EMA34 OR above R3 (reversal signal)
            if close[i] > ema34_1d_aligned[i] or close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h strategy using 1d Camarilla R3/S3 levels as key support/resistance,
# combined with 1d EMA34 trend filter and volume confirmation for breakouts.
# Long when: price > 1d EMA34 (uptrend) AND breaks above R3 resistance with volume spike.
# Short when: price < 1d EMA34 (downtrend) AND breaks below S3 support with volume spike.
# Exits when price crosses back below EMA34 or retests the broken level.
# Camarilla levels provide mathematically derived support/resistance that work well
# in both ranging and trending markets. Volume spike ensures breakout conviction.
# Target: 15-25 trades/year to minimize fee drag while capturing meaningful moves.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).