#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot fade with 1w trend filter and volume confirmation
    # Fade at R3/S3 levels when 1w trend is strong (price > 1w EMA50) and volume spikes
    # Continuation breakout at R4/S4 levels with same filters
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year)
    # Works in both bull and bear markets by using 1w trend filter and volume confirmation
    # Camarilla levels provide clear reversal/breakout levels based on previous day's range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla calculations
    rang = prev_high - prev_low
    camarilla_h5 = prev_close + 1.1 * rang / 6  # R3
    camarilla_h4 = prev_close + 1.1 * rang / 4  # R4
    camarilla_l5 = prev_close - 1.1 * rang / 6  # S3
    camarilla_l4 = prev_close - 1.1 * rang / 4  # S4
    
    # Align 1d Camarilla to 1d (wait for completed 1d bar)
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1w (wait for completed 1w bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h5_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l5_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA50 slope (rising/falling)
        # Use previous bar's EMA to avoid look-ahead
        if i >= 101:
            ema_prev = ema_50_aligned[i-1]
            ema_curr = ema_50_aligned[i]
            ema_rising = ema_curr > ema_prev
            ema_falling = ema_curr < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume confirmation: current 1d volume > 2.0 * 20-period average
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        volume_confirm = vol_1d_aligned[i] > 2.0 * vol_ma_aligned[i]
        
        # Fade signals at R3/S3 (camarilla h5/l5)
        fade_long = close[i] <= l5_aligned[i] and ema_rising and volume_confirm
        fade_short = close[i] >= h5_aligned[i] and ema_falling and volume_confirm
        
        # Breakout signals at R4/S4 (camarilla h4/l4)
        breakout_long = close[i] >= h4_aligned[i] and ema_rising and volume_confirm
        breakout_short = close[i] <= l4_aligned[i] and ema_falling and volume_confirm
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] >= h5_aligned[i] or (not ema_rising and position == 1)
        short_exit = close[i] <= l5_aligned[i] or (not ema_falling and position == -1)
        
        if (fade_long or breakout_long) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (fade_short or breakout_short) and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "6h_1d_1w_camarilla_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0