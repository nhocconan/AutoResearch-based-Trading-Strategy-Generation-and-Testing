#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Enter long when price breaks above 4h Camarilla R3 with volume > 1.8x 20-bar average and price > 1d EMA50 (uptrend).
# Enter short when price breaks below 4h Camarilla S3 with volume > 1.8x 20-bar average and price < 1d EMA50 (downtrend).
# Exit on opposite Camarilla level (R2/S2) to limit drawdown.
# Uses discrete position sizing (0.20) to control risk. Target: 60-150 total trades over 4 years.
# 4h provides signal direction, 1h for entry timing, 1d EMA50 filters counter-trend noise.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.

name = "1h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla calculation (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (using previous bar's OHLC)
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    typical_price = (h_4h + l_4h + c_4h) / 3.0
    hl_range = h_4h - l_4h
    
    r3_4h = typical_price + (hl_range * 1.1 / 4.0)
    s3_4h = typical_price - (hl_range * 1.1 / 4.0)
    r2_4h = typical_price + (hl_range * 1.1 / 6.0)
    s2_4h = typical_price - (hl_range * 1.1 / 6.0)
    
    # Align 4h Camarilla levels to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    r2_4h_aligned = align_htf_to_ltf(prices, df_4h, r2_4h)
    s2_4h_aligned = align_htf_to_ltf(prices, df_4h, s2_4h)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(r2_4h_aligned[i]) or np.isnan(s2_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions with volume confirmation and trend filter
        long_breakout = close[i] > r3_4h_aligned[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i]
        short_breakout = close[i] < s3_4h_aligned[i] and volume_confirm[i] and close[i] < ema_50_1d_aligned[i]
        
        # Exit conditions: opposite Camarilla level (R2/S2)
        long_exit = close[i] < r2_4h_aligned[i]
        short_exit = close[i] > s2_4h_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals