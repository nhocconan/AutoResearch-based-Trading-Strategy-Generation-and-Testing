#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1h volume spike (1.5x median) and 12h EMA50 trend filter.
# Long when price > upper band, 1h volume > 1.5x median volume, and 12h close > 12h EMA50.
# Short when price < lower band, same volume condition, and 12h close < 12h EMA50.
# Exit when price crosses the middle band (mean reversion).
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Combines price channel breakout with volume confirmation and intermediate trend filter for robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian channel (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Get 1h data for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicators: EMA50 trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 1h Indicators: Volume median for spike detection ===
    vol_1h = df_1h['volume'].values
    vol_median_20 = pd.Series(vol_1h).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (4h)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_4h, middle_20)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_median_aligned = align_htf_to_ltf(prices, df_1h, vol_median_20)
    vol_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_1h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20, 50, 20)  # Donchian(20), 1h volume median(20), 12h EMA50
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        middle = middle_20_aligned[i]
        vol_median = vol_median_aligned[i]
        ema_50_12h = ema_50_12h_aligned[i]
        vol_1h = vol_1h_aligned[i]
        
        # Get aligned 12h close for proper trend comparison
        df_12h_close = df_12h['close'].values
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h_close)
        weekly_trend_up = close_12h_aligned[i] > ema_50_12h  # Using 12h close vs 12h EMA50 for trend
        weekly_trend_down = close_12h_aligned[i] < ema_50_12h
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below middle band (mean reversion)
            if price < middle:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above middle band (mean reversion)
            if price > middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 1h volume > 1.5x median volume
            volume_spike = vol_1h > (vol_median * 1.5)
            
            # LONG CONDITIONS
            # Price breaks above upper Donchian band AND volume spike AND 12h uptrend
            if price > upper and volume_spike and weekly_trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below lower Donchian band AND volume spike AND 12h downtrend
            elif price < lower and volume_spike and weekly_trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_1hVolumeSpike1.5x_12hEMA50_v1"
timeframe = "4h"
leverage = 1.0