#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout from 4h with volume confirmation and session filter.
    # Long when price breaks above 4h Camarilla H3 level with volume spike during active session (08-20 UTC).
    # Short when price breaks below 4h Camarilla L3 level with volume spike during active session.
    # Exit when price returns to 4h Camarilla pivot point (mean reversion).
    # Uses 4h for signal direction, 1h for entry timing, and 08-20 UTC session filter to reduce noise.
    # Discrete size 0.20 to minimize fee churn. Target: 60-150 total trades over 4 years (15-37/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h OHLC for Camarilla pivots
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Camarilla pivot levels (based on previous 4h bar)
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # H3 = Close + Range * 1.1 / 4
    # L3 = Close - Range * 1.1 / 4
    
    pivot = (high_4h + low_4h + close_4h) / 3.0
    rng = high_4h - low_4h
    
    # H3 and L3 are the key breakout levels
    camarilla_h3 = close_4h + rng * 1.1 / 4.0
    camarilla_l3 = close_4h - rng * 1.1 / 4.0
    camarilla_pivot = pivot  # Exit level
    
    # Calculate 4h volume mean (20-period) with min_periods
    volume_4h = df_4h['volume'].values
    volume_series = pd.Series(volume_4h)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Entry conditions: price breaks Camarilla H3/L3 levels with volume confirmation
        long_entry = (close[i] > camarilla_h3_aligned[i] and volume_confirmation)
        short_entry = (close[i] < camarilla_l3_aligned[i] and volume_confirmation)
        
        # Exit conditions: price returns to Camarilla pivot point (mean reversion)
        long_exit = close[i] < camarilla_pivot_aligned[i]
        short_exit = close[i] > camarilla_pivot_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_camarilla_breakout_volume_session_v1"
timeframe = "1h"
leverage = 1.0