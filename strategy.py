#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Elder Ray + Volume Spike
# Long when Williams %R(14) < -80 (oversold) + Elder Bull Power > 0 (bullish momentum) + volume > 2x 20-period average
# Short when Williams %R(14) > -20 (overbought) + Elder Bear Power < 0 (bearish momentum) + volume > 2x 20-period average
# Uses 1d HTF for Elder Ray to avoid look-ahead and capture higher timeframe momentum
# Designed for low trade frequency (12-30/year) to minimize fee drag while capturing mean reversion in 6h timeframe
# Williams %R identifies extreme price levels, Elder Ray confirms underlying bull/bear power, volume spike validates conviction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Elder Ray (Bull Power and Bear Power) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA (standard for Elder Ray)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 6h timeframe (completed 1d bar only)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Williams %R(14) calculation on 6h data
        # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().iloc[i]
        lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().iloc[i]
        
        if highest_high_14 == lowest_low_14:
            williams_r = -50.0  # avoid division by zero
        else:
            williams_r = ((highest_high_14 - close[i]) / (highest_high_14 - lowest_low_14)) * -100
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]
        vol_confirm = volume[i] > (vol_sma_20 * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_sma_20)):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R indicates oversold (< -80)
        # 2. Elder Bull Power > 0 (bullish momentum on 1d)
        # 3. Volume confirmation (> 2x average)
        if (williams_r < -80.0) and (bull_power_aligned[i] > 0) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R indicates overbought (> -20)
        # 2. Elder Bear Power < 0 (bearish momentum on 1d)
        # 3. Volume confirmation (> 2x average)
        elif (williams_r > -20.0) and (bear_power_aligned[i] < 0) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_ElderRay_Volume_v1"
timeframe = "6h"
leverage = 1.0