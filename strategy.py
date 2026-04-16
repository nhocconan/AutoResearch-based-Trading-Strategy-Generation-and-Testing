#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 6h Elder Ray Index (Bull/Bear Power) with 1d trend filter (EMA50) and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0, 6h close > 1d EMA50, and 6h volume > 1.3x 20-period median volume.
# Short when Bull Power < 0, Bear Power > 0, 6h close < 1d EMA50, and same volume condition.
# Exit when Elder Ray signals reverse (Bull Power crosses zero or Bear Power crosses zero).
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# This strategy works in both bull and bear markets by using the 1d EMA50 as a regime filter and Elder Ray to measure
# underlying bull/bear power relative to the EMA, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data once before loop for Elder Ray and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # === 6h Indicators: Elder Ray (Bull Power, Bear Power) and volume median ===
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    vol_6h = df_6h['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray (standard period)
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_6h - ema13_6h
    # Bear Power = Low - EMA13
    bear_power = low_6h - ema13_6h
    
    # Calculate 6h volume median (20-period)
    vol_median_20 = pd.Series(vol_6h).rolling(window=20, min_periods=20).median().values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (6h)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    vol_median_aligned = align_htf_to_ltf(prices, df_6h, vol_median_20)
    vol_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_6h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 13, 50)  # volume median(20), EMA13(13), EMA50(1d)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_6h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_6h = vol_6h_aligned[i]
        ema_50_1d = ema_50_1d_aligned[i]
        
        # Get aligned 6h close for trend comparison
        close_6h_aligned = align_htf_to_ltf(prices, df_6h, close_6h)
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when Bull Power crosses below zero (loss of bullish momentum)
            if bull <= 0:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when Bear Power crosses above zero (loss of bearish momentum)
            if bear >= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 6h volume > 1.3x median volume
            volume_spike = vol_6h > (vol_median * 1.3)
            
            # LONG CONDITIONS
            # Bull Power > 0 (bullish momentum), Bear Power < 0 (no bearish momentum),
            # price above 1d EMA50 (uptrend regime), and volume spike
            if bull > 0 and bear < 0 and close_6h_aligned[i] > ema_50_1d and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Bull Power < 0 (no bullish momentum), Bear Power > 0 (bearish momentum),
            # price below 1d EMA50 (downtrend regime), and volume spike
            elif bull < 0 and bear > 0 and close_6h_aligned[i] < ema_50_1d and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA50_6hVolumeSpike1.3x_v1"
timeframe = "6h"
leverage = 1.0