#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Elder Ray + Volume Filter
# Long when: Williams %R(14) < -80 (oversold) + Bull Power > 0 (bullish momentum) + volume > 1.3x 20-period average
# Short when: Williams %R(14) > -20 (overbought) + Bear Power < 0 (bearish momentum) + volume > 1.3x 20-period average
# Uses 6h for entry timing, 1d for Elder Ray (Bull/Bear Power) to capture institutional momentum
# Designed for low trade frequency (12-30/year) with high win rate in mean-reverting conditions
# Works in both bull and bear markets by fading extremes with momentum confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h and 1d HTF data once before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_6h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 6h Indicators: Williams %R(14) ===
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Williams %R calculation: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # === 1d Indicators: Elder Ray (Bull Power and Bear Power) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema_13
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema_13
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R indicates oversold (< -80)
        # 2. Bull Power > 0 (bullish momentum)
        # 3. Volume confirmation
        if (williams_r_aligned[i] < -80) and (bull_power_aligned[i] > 0) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R indicates overbought (> -20)
        # 2. Bear Power < 0 (bearish momentum)
        # 3. Volume confirmation
        elif (williams_r_aligned[i] > -20) and (bear_power_aligned[i] < 0) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_ElderRay_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0