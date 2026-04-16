#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian channel breakout (20) with 12h EMA trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20) AND 12h EMA50 > price AND volume > 1.5x average volume.
# Short when price breaks below lower Donchian(20) AND 12h EMA50 < price AND volume > 1.5x average volume.
# Exit when price crosses back inside Donchian channel or volume drops below average.
# Uses discrete position size 0.25. Donchian captures trends, EMA filter avoids counter-trend trades, volume confirms momentum.
# 4h timeframe targets 20-50 trades/year to minimize fee drag.
# Works in bull markets (catch uptrend breakouts) and bear markets (catch downtrend breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Get 12h data once before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Upper channel: highest high over 20 periods
    # Lower channel: lowest low over 20 periods
    def rolling_max(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_20 = rolling_max(high_4h, 20)
    lower_20 = rolling_min(low_4h, 20)
    
    # === 12h Indicators: EMA50 for trend filter ===
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume: Average volume (20-period) for confirmation ===
    def rolling_mean(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    avg_volume_20 = rolling_mean(volume, 20)
    
    # Align all indicators to primary timeframe (4h)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    avg_volume_aligned = align_htf_to_ltf(prices, df_4h, avg_volume_20)  # volume is already 4h, but align for consistency
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100  # Donchian20 needs sufficient warmup + EMA50 + volume average
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(avg_volume_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        ema50 = ema50_aligned[i]
        vol_avg = avg_volume_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price crosses below upper Donchian OR volume drops below average
            if (price < upper) or (vol < vol_avg):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price crosses above lower Donchian OR volume drops below average
            if (price > lower) or (vol < vol_avg):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper Donchian AND price > 12h EMA50 (uptrend) AND volume > 1.5x average
            if (price > upper) and (price > ema50) and (vol > 1.5 * vol_avg):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower Donchian AND price < 12h EMA50 (downtrend) AND volume > 1.5x average
            elif (price < lower) and (price < ema50) and (vol > 1.5 * vol_avg):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeSpike_12hEMA50_TrendFilter_V1"
timeframe = "4h"
leverage = 1.0