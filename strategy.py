#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation.
# Long when price breaks above Donchian upper (20-period high) and close > 1d EMA50 and volume > 1.5x 20-period median volume.
# Short when price breaks below Donchian lower (20-period low) and close < 1d EMA50 and same volume condition.
# Exit when price crosses the opposite Donchian band (long exits at lower band, short exits at upper band).
# Uses discrete position size 0.30. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide clear structure, EMA50 filters regime, volume confirms breakout strength.
# Works in bull markets via breakouts, in bear markets via short breakdowns, avoids whipsaws via regime filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian(20) and volume median ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    vol_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian bands (20-period)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume median (20-period)
    vol_median_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).median().values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 50)  # Donchian(20), EMA50(1d)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_4h = vol_4h_aligned[i]
        ema_50_1d = ema_50_1d_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Donchian lower band
            if price < lower:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Donchian upper band
            if price > upper:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 4h volume > 1.5x median volume
            volume_spike = vol_4h > (vol_median * 1.5)
            
            # LONG CONDITIONS
            # Price breaks above Donchian upper, price above 1d EMA50 (uptrend regime), and volume spike
            if price > upper and close > ema_50_1d and volume_spike:
                signals[i] = 0.30
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower, price below 1d EMA50 (downtrend regime), and volume spike
            elif price < lower and close < ema_50_1d and volume_spike:
                signals[i] = -0.30
                position = -1
        
        else:
            signals[i] = position * 0.30  # maintain position
    
    return signals

name = "4h_Donchian20_1dEMA50_4hVolumeSpike1.5x_v1"
timeframe = "4h"
leverage = 1.0