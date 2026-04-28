#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla H4/L4 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Uses 6h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years).
# 12h EMA50 provides primary trend filter: bull when price > EMA50, bear when price < EMA50.
# Camarilla H4/L4 from 1d provide institutional pivot points with proven edge.
# Volume spike (>2.0x 24-bar average) confirms breakout strength.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.

name = "6h_Camarilla_H4L4_Breakout_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to reduce noise
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivots (H4, L4) and 12h data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 4 or len(df_12h) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_12h = df_12h['close'].values
    
    # Calculate 1d Camarilla pivot levels (H4, L4)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h4_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0  # H4 = Close + 1.1*(Range)/2
    l4_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0  # L4 = Close - 1.1*(Range)/2
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h volume spike: >2.0x 24-bar average volume (accounting for session gaps)
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(h4_1d_aligned[i]) or
            np.isnan(l4_1d_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA50 direction (price above/below EMA50)
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > h4_1d_aligned[i]
        short_breakout = close[i] < l4_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: opposite Camarilla level (L4/H4 for reversion)
        long_exit = close[i] < l4_1d_aligned[i]  # Exit long at L4
        short_exit = close[i] > h4_1d_aligned[i]  # Exit short at H4
        
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