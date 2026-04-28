#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla R4/S4 levels with 1d EMA50 trend filter and volume confirmation (>2x average).
# Enter long when price breaks above weekly Camarilla R4 level with volume > 2x average and close > 1d EMA50 (bullish bias).
# Enter short when price breaks below weekly Camarilla S4 level with volume > 2x average and close < 1d EMA50 (bearish bias).
# Exit when price returns to the weekly Camarilla midpoint (P) or touches the opposite level (S4 for long exit, R4 for short exit).
# Uses discrete position sizing (0.25) to control risk and minimize fee churn. Target: 50-150 total trades over 4 years.
# Weekly Camarilla provides stable structure; 1d EMA50 filters counter-trend breakouts; volume confirms institutional interest.
# Designed to work in both bull (breakouts with trend) and bear (breakdowns with trend) markets by aligning with HTF direction.

name = "6h_Camarilla_R4S4_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation (MTF structure)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (using previous bar's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True range for Camarilla calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w)
    tr3 = np.abs(low_1w - close_1w)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Camarilla levels (based on previous bar's close and range)
    camarilla_pivot = close_1w  # Pivot is previous close
    camarilla_range = high_1w - low_1w
    
    # R4 and S4 levels (strongest breakout levels for very few trades)
    r4 = camarilla_pivot + camarilla_range * 1.1 / 2
    s4 = camarilla_pivot - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Get 1d data for EMA50 trend filter (HTF trend)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d EMA50 bias
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        # Weekly Camarilla breakout conditions
        long_breakout = close[i] > r4_aligned[i]
        short_breakout = close[i] < s4_aligned[i]
        
        # Exit conditions: return to pivot or touch opposite level
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