#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
# Enter long when price breaks above 12h Camarilla R1 level with volume > 1.8x average and close > 1d EMA50 (bullish bias).
# Enter short when price breaks below 12h Camarilla S1 level with volume > 1.8x average and close < 1d EMA50 (bearish bias).
# Exit when price returns to the 12h Camarilla midpoint (P) or opposite level (S1 for long exit, R1 for short exit).
# Uses Camarilla structure for pivot points, higher timeframe EMA for trend filter, and volume for confirmation.
# Works in bull markets (breakouts continue up with trend) and bear markets (breakdowns continue down with trend).
# Uses discrete position sizing (0.25) to control risk. Target: 75-200 total trades over 4 years.

name = "4h_Camarilla_R1S1_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (using previous bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # True range for Camarilla calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - close_12h)
    tr3 = np.abs(low_12h - close_12h)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Camarilla levels (based on previous bar's close and range)
    camarilla_pivot = close_12h  # Pivot is previous close
    camarilla_range = high_12h - low_12h
    
    # R1 and S1 levels (more frequently touched than R3/S3)
    r1 = camarilla_pivot + camarilla_range * 1.1 / 12
    s1 = camarilla_pivot - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d EMA50 bias
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > r1_aligned[i]
        short_breakout = close[i] < s1_aligned[i]
        
        # Exit conditions: return to pivot or opposite level touched
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_bias
        short_entry = short_breakout and vol_confirm and bearish_bias
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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