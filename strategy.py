#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w EMA trend filter.
# Long when price breaks above 4h Donchian upper(20) AND 1d volume > 1.8x 20-period average AND price > 1w EMA(50).
# Short when price breaks below 4h Donchian lower(20) AND 1d volume > 1.8x 20-period average AND price < 1w EMA(50).
# Exit when price returns to 4h Donchian midpoint.
# Uses discrete position size 0.25. Donchian provides structure, volume confirmation reduces false signals,
# and 1w EMA ensures we only trade with the higher timeframe trend. Target: 80-150 total trades over 4 years (20-38/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # === 4h Indicators: Donchian channels (20-period) ===
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Align Donchian levels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    middle_aligned = align_htf_to_ltf(prices, df_4h, middle_20)
    
    # Get 1d data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume spike filter ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 1w data once before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA(50) for trend filter ===
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        middle_val = middle_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        ema_50_val = ema_50_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.8x 20-period average (using 1d volume MA)
        vol_filter = vol > 1.8 * vol_ma_val if vol_ma_val > 0 else False
        
        # Trend filter: price relative to 1w EMA(50)
        trend_filter_long = price > ema_50_val
        trend_filter_short = price < ema_50_val
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Donchian middle
            if price <= middle_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Donchian middle
            if price >= middle_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian upper with volume and trend confirmation
            if price > upper_val and vol_filter and trend_filter_long:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price breaks below Donchian lower with volume and trend confirmation
            elif price < lower_val and vol_filter and trend_filter_short:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_1wEMA50Trend_V1"
timeframe = "4h"
leverage = 1.0