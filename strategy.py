# Solution
#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    trend_down = close < ema34_1d_aligned
    
    # 12h Camarilla pivot levels (based on previous day's range)
    # We'll calculate pivot levels using 1d data but align to 12h
    # Camarilla levels: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # We use the previous day's range for today's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_pivot = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    rng = high_1d - low_1d
    r3 = close_1d_for_pivot + 1.1 * rng
    s3 = close_1d_for_pivot - 1.1 * rng
    r4 = close_1d_for_pivot + 1.5 * rng
    s4 = close_1d_for_pivot - 1.5 * rng
    
    # Align to 12h timeframe (1 bar = 12h, so 2 bars per day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is valid
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation and 1d uptrend
            if close[i] > r3_aligned[i] and vol_confirm[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume confirmation and 1d downtrend
            elif close[i] < s3_aligned[i] and vol_confirm[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 or trend turns down
            if close[i] < s3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or trend turns up
            if close[i] > r3_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with volume confirmation and 1d trend filter
# Captures institutional breakout patterns in both bull and bear markets.
# Long when price breaks above R3 (bullish breakout) in 1d uptrend with volume.
# Short when price breaks below S3 (bearish breakdown) in 1d downtrend with volume.
# Uses 12h timeframe for optimal trade frequency (12-37 trades/year target).
# Volume confirmation reduces false breakouts. Trend filter ensures alignment with higher timeframe.
# Position size 0.25 manages risk through volatile crypto cycles.