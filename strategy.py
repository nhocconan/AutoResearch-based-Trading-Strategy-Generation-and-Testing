#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h Supertrend + volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (13-period EMA on 6h)
# - Supertrend on 12h: trend filter (ATR=10, mult=3.0) - only take longs in uptrend, shorts in downtrend
# - Volume confirmation: 6h volume > 1.5x 20-period average volume to avoid low-participation signals
# - Works in bull/bear: Supertrend regime filter adapts to market direction, Elder Ray measures momentum strength
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines

name = "6h_12h_elderray_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h Supertrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr_12h1[0] if 'tr_12h1' in locals() else tr_12h[0]  # handle first element
    tr_12h1 = high_12h - low_12h  # recompute for clarity
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    
    atr_12h = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_12h = (high_12h + low_12h) / 2
    upper_band_12h = hl2_12h + (3.0 * atr_12h)
    lower_band_12h = hl2_12h - (3.0 * atr_12h)
    
    supertrend_12h = np.zeros_like(close_12h)
    direction_12h = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > upper_band_12h[i-1]:
            direction_12h[i] = 1
        elif close_12h[i] < lower_band_12h[i-1]:
            direction_12h[i] = -1
        else:
            direction_12h[i] = direction_12h[i-1]
            if direction_12h[i] == 1 and lower_band_12h[i] < lower_band_12h[i-1]:
                lower_band_12h[i] = lower_band_12h[i-1]
            if direction_12h[i] == -1 and upper_band_12h[i] > upper_band_12h[i-1]:
                upper_band_12h[i] = upper_band_12h[i-1]
    
        supertrend_12h[i] = lower_band_12h[i] if direction_12h[i] == 1 else upper_band_12h[i]
    
    supertrend_dir_12h = direction_12h  # 1=uptrend, -1=downtrend
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, supertrend_dir_12h)
    
    # Pre-compute 6h Elder Ray (Bull/Bear Power) with 13-period EMA
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low_6h   # Bear Power = EMA13 - Low
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_6h > (1.5 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, prices, volume_spike)  # same timeframe
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(supertrend_dir_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power becomes stronger than Bull Power OR Supertrend turns down
            if bear_power[i] > bull_power[i] or supertrend_dir_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power becomes stronger than Bear Power OR Supertrend turns up
            if bull_power[i] > bear_power[i] or supertrend_dir_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with volume spike and Supertrend alignment
            if volume_spike_aligned[i]:
                # Long: Bull Power > Bear Power (bullish momentum) AND Supertrend uptrend
                if bull_power[i] > bear_power[i] and supertrend_dir_aligned[i] == 1:
                    position = 1
                    signals[i] = 0.25
                # Short: Bear Power > Bull Power (bearish momentum) AND Supertrend downtrend
                elif bear_power[i] > bull_power[i] and supertrend_dir_aligned[i] == -1:
                    position = -1
                    signals[i] = -0.25
    
    return signals