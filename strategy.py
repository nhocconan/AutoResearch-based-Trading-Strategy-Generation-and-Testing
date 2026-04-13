#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian breakout with 1w trend filter and volume confirmation.
    # Works in bull markets (breakouts up) and bear markets (breakouts down) by following the 1w trend.
    # Uses discrete position size 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels and volume (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period) based on previous bar (no look-ahead)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Upper band: highest high of previous 20 bars
    upper_20 = np.full_like(high_6h, np.nan)
    for i in range(20, len(high_6h)):
        upper_20[i] = np.max(high_6h[i-20:i])
    
    # Lower band: lowest low of previous 20 bars
    lower_20 = np.full_like(low_6h, np.nan)
    for i in range(20, len(low_6h)):
        lower_20[i] = np.min(low_6h[i-20:i])
    
    # Calculate 6h volume mean (20-period) with min_periods
    volume_6h_series = pd.Series(df_6h['volume'].values)
    vol_ma_20_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA (21-period) with min_periods
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Align HTF indicators to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_6h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_6h, lower_20)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h volume for spike detection
        volume_6h_raw = df_6h['volume'].values
        vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h_raw)
        
        # Volume filter: current 6h volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_6h_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: price above/below 1w EMA indicates trend direction
        trend_up = close[i] > ema_aligned[i]
        trend_down = close[i] < ema_aligned[i]
        
        # Entry conditions: price breaks Donchian channel with volume confirmation and trend filter
        long_entry = (close[i] > upper_aligned[i] and volume_confirmation and trend_up)
        short_entry = (close[i] < lower_aligned[i] and volume_confirmation and trend_down)
        
        # Exit conditions: price returns to opposite Donchian band (mean reversion exit)
        long_exit = close[i] < lower_aligned[i]
        short_exit = close[i] > upper_aligned[i]
        
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

name = "6h_6h_1w_donchian_breakout_volume_ema_v1"
timeframe = "6h"
leverage = 1.0