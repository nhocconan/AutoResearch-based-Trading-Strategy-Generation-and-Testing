# 12h_1w_1d_camarilla_breakout_volume
# Hypothesis: 12-hour trading using weekly and daily timeframe confluence. 
# Uses weekly high/low for major trend bias and daily Camarilla levels for precise entries.
# Volume confirmation filters false breakouts. Designed for lower frequency (12-37 trades/year)
# to minimize fee drag while capturing major moves in both bull and bear markets.

name = "12h_1w_1d_camarilla_breakout_volume"
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
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly trend bias: price above/below weekly midpoint
    weekly_mid = (high_1w + low_1w) / 2
    weekly_bias_above = close_1w > weekly_mid
    weekly_bias_below = close_1w < weekly_mid
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first value
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    # Resistance levels
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    # Support levels
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # Align weekly bias to 12h timeframe
    weekly_bias_above_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_above.astype(float))
    weekly_bias_below_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_below.astype(float))
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(weekly_bias_above_aligned[i]) or np.isnan(weekly_bias_below_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: weekly bullish bias + price breaks above R4 with volume
        if (weekly_bias_above_aligned[i] > 0.5 and 
            close[i] > r4_aligned[i] and vol_confirm[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: weekly bearish bias + price breaks below S4 with volume
        elif (weekly_bias_below_aligned[i] > 0.5 and 
              close[i] < s4_aligned[i] and vol_confirm[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals