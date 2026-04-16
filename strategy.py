#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy using 4h Donchian(20) breakout with volume confirmation and 1d EMA200 trend filter.
# Long when price > 4h upper Donchian band, 1h volume > 1.5x median, and 1d close > 1d EMA200.
# Short when price < 4h lower Donchian band, same volume condition, and 1d close < 1d EMA200.
# Exit when price crosses the 4h middle Donchian band.
# Uses discrete position size 0.20. Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 60-150 total trades over 4 years (15-37/year). Uses 4h/1d for direction, 1h for timing.

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
    
    # === 1h Indicators: Volume median for spike detection ===
    vol_1h = df_1h['volume'].values
    vol_median_20 = pd.Series(vol_1h).rolling(window=20, min_periods=20).median().values
    
    # Get 1d data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === 1d Indicators: EMA200 trend filter ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all indicators to primary timeframe (1h)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_4h, middle_20)
    vol_median_aligned = align_htf_to_ltf(prices, df_1h, vol_median_20)
    vol_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_1h)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20, 200, 20)  # Donchian(20), 1h volume median(20), 1d EMA200
    
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
            np.isnan(vol_1h_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        middle = middle_20_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_1h = vol_1h_aligned[i]
        ema_200_1d = ema_200_1d_aligned[i]
        
        # Get aligned 1d close for proper trend comparison
        df_1d_close = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d_close)
        daily_trend_up = close_1d_aligned[i] > ema_200_1d  # Using 1d close vs 1d EMA200 for trend
        daily_trend_down = close_1d_aligned[i] < ema_200_1d
        
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
            # Volume spike filter: current 1h volume > 1.5x median volume
            volume_spike = vol_1h > (vol_median * 1.5)
            
            # LONG CONDITIONS
            # Price breaks above upper Donchian band AND volume spike AND 1d uptrend
            if price > upper and volume_spike and daily_trend_up:
                signals[i] = 0.20
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below lower Donchian band AND volume spike AND 1d downtrend
            elif price < lower and volume_spike and daily_trend_down:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20  # maintain position
    
    return signals

name = "1h_Donchian20_1hVolumeSpike1.5x_1dEMA200_v1"
timeframe = "1h"
leverage = 1.0