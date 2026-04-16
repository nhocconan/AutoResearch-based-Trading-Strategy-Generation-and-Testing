#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Donchian channel breakout (20-period) with 1d trend filter (EMA50) and volume confirmation.
# Long when price breaks above upper Donchian(20) AND close > 1d EMA50 AND volume > 1.5x 20-period median volume.
# Short when price breaks below lower Donchian(20) AND close < 1d EMA50 AND volume > 1.5x 20-period median volume.
# Exit when price returns to the middle of the Donchian channel (mean reversion) or opposite breakout occurs.
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# This strategy captures strong momentum breaks in both bull and bear markets, with 1d EMA50 filtering for regime alignment
# and volume confirmation reducing false breakouts. Donchian exit provides systematic profit-taking and loss limitation.

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
    dc_upper = pd.Series(high_12h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_lower = pd.Series(low_12h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    dc_middle = (dc_upper + dc_lower) / 2.0
    
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
    dc_upper_aligned = align_htf_to_ltf(prices, df_12h, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_12h, dc_lower)
    dc_middle_aligned = align_htf_to_ltf(prices, df_12h, dc_middle)
    vol_median_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20)
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_window, 20, 50)  # Donchian(20), volume median(20), EMA50(1d)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper_aligned[i]) or np.isnan(dc_lower_aligned[i]) or 
            np.isnan(dc_middle_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(vol_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        dc_upper = dc_upper_aligned[i]
        dc_lower = dc_lower_aligned[i]
        dc_middle = dc_middle_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_12h = vol_12h_aligned[i]
        ema_50_1d = ema_50_1d_aligned[i]
        
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price returns to middle of Donchian channel (mean reversion)
            if price <= dc_middle:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price returns to middle of Donchian channel (mean reversion)
            if price >= dc_middle:
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
            # Price breaks above upper Donchian AND price above 1d EMA50 AND volume spike
            if price > dc_upper and close > ema_50_1d and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below lower Donchian AND price below 1d EMA50 AND volume spike
            elif price < dc_lower and close < ema_50_1d and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Donchian20_1dEMA50_12hVolumeSpike1.5x_v1"
timeframe = "12h"
leverage = 1.0