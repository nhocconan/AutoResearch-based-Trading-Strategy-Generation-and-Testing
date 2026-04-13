#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation.
    # Camarilla pivots from 1d provide institutional support/resistance levels.
    # Breakout above R4 or below S4 with 1w trend alignment and volume spike indicates strong continuation.
    # Works in both bull and bear markets via trend filter. Target: 50-150 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP)
    PP = (high_1d + low_1d + close_1d) / 3
    # Range
    RANGE = high_1d - low_1d
    # Camarilla levels
    R4 = PP + RANGE * 1.1 / 2
    S4 = PP - RANGE * 1.1 / 2
    R3 = PP + RANGE * 1.1 / 4
    S3 = PP - RANGE * 1.1 / 4
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h volume mean (20-period) with min_periods
    df_6h = get_htf_data(prices, '6h')
    volume_6h_series = pd.Series(df_6h['volume'].values)
    vol_ma_20_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h volume for spike detection
        volume_6h_raw = df_6h['volume'].values
        vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h_raw)
        
        # Volume filter: current 6h volume > 2.0 * 20-period mean (volume spike for confirmation)
        volume_confirmation = vol_6h_aligned[i] > 2.0 * vol_ma_aligned[i]
        
        # Trend filter: price above/below weekly EMA50
        price_above_weekly_ema = close[i] > ema_50_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_aligned[i]
        
        # Camarilla breakout conditions
        breakout_long = close[i] > R4_aligned[i]  # Break above R4
        breakout_short = close[i] < S4_aligned[i]  # Break below S4
        
        # Entry conditions: breakout with volume spike and trend alignment
        long_entry = breakout_long and volume_confirmation and price_above_weekly_ema
        short_entry = breakout_short and volume_confirmation and price_below_weekly_ema
        
        # Exit conditions: price returns to R3/S3 levels or loss of volume confirmation
        long_exit = close[i] < R3_aligned[i] or not volume_confirmation
        short_exit = close[i] > S3_aligned[i] or not volume_confirmation
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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

name = "6h_1d_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0