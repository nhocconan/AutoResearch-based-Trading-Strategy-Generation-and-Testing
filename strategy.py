#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H4/L4 breakout with 1w EMA50 trend filter and volume confirmation.
# Uses 12h primary timeframe to reduce trade frequency while capturing multi-week moves.
# Camarilla H4/L4 levels from 1w provide strong weekly support/resistance, filtered by 1w EMA50 trend and volume spikes.
# Designed to work in both bull and bear markets by following the 1w trend while using Camarilla levels as entry signals.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

name = "12h_Camarilla_H4L4_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for Camarilla levels (H4, L4) and EMA50 (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels (H4, L4)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    H4 = close_1w + range_1w * 1.1 / 2
    L4 = close_1w - range_1w * 1.1 / 2
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w indicators to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # EMA50 needs 50 bars, volume MA needs 20, use 50 for safety
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(H4_aligned[i]) or
            np.isnan(L4_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA50 direction
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > H4_aligned[i]
        short_breakout = close[i] < L4_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Calculate 1w Camarilla H6/L6 levels for exit
        H6 = close_1w + range_1w * 1.1
        L6 = close_1w - range_1w * 1.1
        
        H6_aligned = align_htf_to_ltf(prices, df_1w, H6)
        L6_aligned = align_htf_to_ltf(prices, df_1w, L6)
        
        long_exit = close[i] < H6_aligned[i]
        short_exit = close[i] > L6_aligned[i]
        
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