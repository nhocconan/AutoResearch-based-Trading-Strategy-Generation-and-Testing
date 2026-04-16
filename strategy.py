#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with 12h volume confirmation and 1d EMA50 trend filter.
# Long when price > 12h upper Donchian, 12h volume > 1.5x 20-period median, and 1d close > 1d EMA50.
# Short when price < 12h lower Donchian, same volume condition, and 1d close < 1d EMA50.
# Exit when price crosses the 12h middle Donchian band.
# Uses discrete position size 0.25. Session filter: 08-20 UTC.
# Target: 75-200 total trades over 4 years (19-50/year). Uses 12h for price channel, 1d for trend.

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
    
    # Get 12h data once before loop for Donchian levels and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian channel (20-period) and volume median ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate Donchian channels
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Calculate 12h volume median (20-period)
    vol_median_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).median().values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA50 for trend ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (4h)
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_12h, middle_20)
    vol_median_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20)
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 50)  # Donchian(20), EMA50(1d)
    
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
            np.isnan(middle_20_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(vol_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        middle = middle_20_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_12h = vol_12h_aligned[i]
        ema_50_1d = ema_50_1d_aligned[i]
        
        # Get aligned 1d close for proper trend comparison
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        daily_trend_up = close_1d_aligned[i] > ema_50_1d
        daily_trend_down = close_1d_aligned[i] < ema_50_1d
        
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
            # Volume spike filter: current 12h volume > 1.5x median volume
            volume_spike = vol_12h > (vol_median * 1.5)
            
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

name = "4h_Donchian20_12hVolumeSpike1.5x_1dEMA50_v1"
timeframe = "4h"
leverage = 1.0