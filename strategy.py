#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
    # Works in bull markets (breakouts up) and bear markets (breakouts down) by following the 12h trend.
    # Uses discrete position size 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels and volume (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) based on previous bar (no look-ahead)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high of previous 20 bars
    upper_20 = np.full_like(high_4h, np.nan)
    for i in range(20, len(high_4h)):
        upper_20[i] = np.max(high_4h[i-20:i])
    
    # Lower band: lowest low of previous 20 bars
    lower_20 = np.full_like(low_4h, np.nan)
    for i in range(20, len(low_4h)):
        lower_20[i] = np.min(low_4h[i-20:i])
    
    # Calculate 4h volume mean (20-period) with min_periods
    volume_4h_series = pd.Series(df_4h['volume'].values)
    vol_ma_20_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA (21-period) with min_periods
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Align HTF indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h volume for spike detection
        volume_4h_raw = df_4h['volume'].values
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h_raw)
        
        # Volume filter: current 4h volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_4h_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: price above/below 12h EMA indicates trend direction
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

name = "4h_4h_12h_donchian_breakout_volume_ema_v1"
timeframe = "4h"
leverage = 1.0