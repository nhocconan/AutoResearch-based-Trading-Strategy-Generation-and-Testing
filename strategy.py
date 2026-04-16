#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter.
# Long when price > 1d upper Donchian band, 12h volume > 1.3x median, and 1d close > 1d EMA50.
# Short when price < 1d lower Donchian band, same volume condition, and 1d close < 1d EMA50.
# Exit when price crosses the 1d middle Donchian band.
# Uses discrete position size 0.25. Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 50-150 total trades over 4 years (12-37/year). Uses 1d for direction/structure, 12h for timing.

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
    
    # Get 1d data once before loop for Donchian levels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian channel (20-period) and EMA50 ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Calculate EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Volume median for spike detection ===
    vol_12h = df_12h['volume'].values
    vol_median_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (12h)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_1d, middle_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_median_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20)
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 50, 20)  # Donchian(20), EMA50, 12h volume median(20)
    
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
            np.isnan(middle_20_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        middle = middle_20_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_12h = vol_12h_aligned[i]
        
        # Get aligned 1d close for proper trend comparison
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        daily_trend_up = close_1d_aligned[i] > ema_50  # Using 1d close vs 1d EMA50 for trend
        daily_trend_down = close_1d_aligned[i] < ema_50
        
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
            # Volume spike filter: current 12h volume > 1.3x median volume
            volume_spike = vol_12h > (vol_median * 1.3)
            
            # LONG CONDITIONS
            # Price breaks above upper Donchian band AND volume spike AND 1d uptrend
            if price > upper and volume_spike and daily_trend_up:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below lower Donchian band AND volume spike AND 1d downtrend
            elif price < lower and volume_spike and daily_trend_down:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Donchian20_12hVolumeSpike1.3x_1dEMA50_v1"
timeframe = "12h"
leverage = 1.0