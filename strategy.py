#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Donchian channel breakout (20-period) with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band, 12h close > 1d EMA50, and 12h volume > 1.5x 20-period median volume.
# Short when price breaks below Donchian lower band, 12h close < 1d EMA50, and same volume condition.
# Exit when price touches the opposite Donchian band (upper for shorts, lower for longs).
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# This strategy works in both bull and bear markets by using the 1d EMA50 as a regime filter and Donchian breakouts
# to capture strong directional moves, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian Channel (20-period) and volume median ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian Channel (20-period)
    donchian_window = 20
    upper_12h = pd.Series(high_12h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_12h = pd.Series(low_12h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate 12h volume median (20-period)
    vol_median_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).median().values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (12h)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    vol_median_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20)
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_window, 20, 50)  # Donchian(20), volume median(20), EMA50(1d)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_12h_aligned[i]) or 
            np.isnan(close_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_12h_aligned[i]
        lower = lower_12h_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_12h = vol_12h_aligned[i]
        price_12h = close_12h_aligned[i]
        ema_50_1d = ema_50_1d_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price touches or goes below Donchian lower band
            if price <= lower:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price touches or goes above Donchian upper band
            if price >= upper:
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
            # Price breaks above Donchian upper band, price above 1d EMA50 (uptrend regime), and volume spike
            if price > upper and price_12h > ema_50_1d and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower band, price below 1d EMA50 (downtrend regime), and volume spike
            elif price < lower and price_12h < ema_50_1d and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Donchian20_1dEMA50_12hVolumeSpike1.5x_v1"
timeframe = "12h"
leverage = 1.0