#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
# Uses 1d primary timeframe targeting 7-25 trades/year (30-100 total over 4 years).
# 1w EMA50 provides primary trend filter: bull when close > EMA50, bear when close < EMA50.
# Camarilla H3/L3 from 1w provide institutional pivot points with proven edge.
# Volume confirmation (>1.5x 20-bar average) ensures breakout strength.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn. Works in both bull and bear via trend filter.

name = "1d_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivots (H3, L3) and 1w data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels (H3, L3)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    h3_1w = close_1w + (high_1w - low_1w) * 1.1 / 4.0  # H3 = Close + 1.1*(Range)/4
    l3_1w = close_1w - (high_1w - low_1w) * 1.1 / 4.0  # L3 = Close - 1.1*(Range)/4
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1w_aligned[i]) or
            np.isnan(l3_1w_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA50 direction (close above/below EMA50)
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > h3_1w_aligned[i]
        short_breakout = close[i] < l3_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: opposite Camarilla level (L3/H3 for reversion)
        long_exit = close[i] < l3_1w_aligned[i]  # Exit long at L3
        short_exit = close[i] > h3_1w_aligned[i]  # Exit short at H3
        
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