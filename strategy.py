#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with 1d volume confirmation and 1w EMA50 trend filter.
# Long when price > 1w upper Donchian band, 1d volume > 1.5x median, and 1w close > 1w EMA50.
# Short when price < 1w lower Donchian band, same volume condition, and 1w close < 1w EMA50.
# Exit when price crosses the 1w middle Donchian band.
# Uses discrete position size 0.25. Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 30-100 total trades over 4 years (7-25/year). Uses 1w for direction/structure, 1d for timing/volume.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1w data once before loop for Donchian levels and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: Donchian channel (20-period) and EMA50 ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Calculate EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Volume median for spike detection ===
    vol_1d = df_1d['volume'].values
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (1d)
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_1w, middle_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 50, 20)  # Donchian(20), EMA50, 1d volume median(20)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        middle = middle_20_aligned[i]
        ema_50 = ema_50_1w_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_1d = vol_1d_aligned[i]
        
        # Get aligned 1w close for proper trend comparison
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        weekly_trend_up = close_1w_aligned[i] > ema_50  # Using 1w close vs 1w EMA50 for trend
        weekly_trend_down = close_1w_aligned[i] < ema_50
        
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
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 1d volume > 1.5x median volume
            volume_spike = vol_1d > (vol_median * 1.5)
            
            # LONG CONDITIONS
            # Price breaks above upper Donchian band AND volume spike AND 1w uptrend
            if price > upper and volume_spike and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below lower Donchian band AND volume spike AND 1w downtrend
            elif price < lower and volume_spike and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_Donchian20_1dVolumeSpike1.5x_1wEMA50_v1"
timeframe = "1d"
leverage = 1.0