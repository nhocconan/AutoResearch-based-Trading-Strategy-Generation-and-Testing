#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Camarilla H3/L3 levels with 1d EMA50 trend filter and volume confirmation.
# Enter long when price breaks above 1w Camarilla H3 level with volume > 1.8x average and close > 1d EMA50 (bullish bias).
# Enter short when price breaks below 1w Camarilla L3 level with volume > 1.8x average and close < 1d EMA50 (bearish bias).
# Exit when price returns to the 1w Camarilla P level.
# Uses weekly structure for key pivot levels, daily EMA for trend filter, and volume for confirmation.
# Designed for lower trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
# Works in bull markets (breakouts continue up with trend) and bear markets (breakdowns continue down with trend).

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1w data for Camarilla pivot calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w Camarilla levels (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # True range for Camarilla calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w)
    tr3 = np.abs(low_1w - close_1w)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Camarilla levels (based on previous week's close and range)
    camarilla_pivot = close_1w  # Pivot is previous close
    camarilla_range = high_1w - low_1w
    
    # H3 and L3 levels (Camarilla: H4 = P + 1.1/2*range, H3 = P + 1.1/4*range, L3 = P - 1.1/4*range)
    r3 = camarilla_pivot + camarilla_range * 1.1 / 4
    s3 = camarilla_pivot - camarilla_range * 1.1 / 4
    pivot = camarilla_pivot
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume confirmation: >1.8x 30-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_30 = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > 1.8 * volume_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d EMA50 bias
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > h3_aligned[i]
        short_breakout = close[i] < l3_aligned[i]
        
        # Exit condition: return to pivot
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