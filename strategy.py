#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Enter long when price breaks above 1w Camarilla R3 with volume > 2.0x 50-bar average and price > 1d EMA50 (uptrend).
# Enter short when price breaks below 1w Camarilla S3 with volume > 2.0x 50-bar average and price < 1d EMA50 (downtrend).
# Exit on opposite Camarilla level (R2/S2) to limit drawdown.
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.
# 1w Camarilla provides stronger structural levels than 12h, reducing false breakouts. 1d EMA50 filters counter-trend noise.
# Volume confirmation ensures breakout strength. Works in bull (breakouts with trend) and bear (failed breaks via exits) markets.

name = "12h_Camarilla_R3S3_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w Camarilla levels (using previous week's OHLC)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    typical_price = (h_1w + l_1w + c_1w) / 3.0
    hl_range = h_1w - l_1w
    
    r3_1w = typical_price + (hl_range * 1.1 / 4.0)
    s3_1w = typical_price - (hl_range * 1.1 / 4.0)
    r2_1w = typical_price + (hl_range * 1.1 / 6.0)
    s2_1w = typical_price - (hl_range * 1.1 / 6.0)
    
    # Align 1w Camarilla levels to 12h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume confirmation: >2.0x 50-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_50 = volume_series.rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > 2.0 * volume_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions with volume confirmation and trend filter
        long_breakout = close[i] > r3_1w_aligned[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i]
        short_breakout = close[i] < s3_1w_aligned[i] and volume_confirm[i] and close[i] < ema_50_1d_aligned[i]
        
        # Exit conditions: opposite Camarilla level (R2/S2)
        long_exit = close[i] < r2_1w_aligned[i]
        short_exit = close[i] > s2_1w_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals