#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA trend filter and volume confirmation.
# Uses 1d EMA(50) for trend bias and Alligator convergence/divergence for entry timing.
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3) - measures market sleeping/awakening.
# Long when Lips > Teeth > Jaw (bullish alignment) + price above 1d EMA50 + volume confirmation.
# Short when Lips < Teeth < Jaw (bearish alignment) + price below 1d EMA50 + volume confirmation.
# Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag.
# Works in bull/bear: 1d EMA filter avoids counter-trend trades, Alligator captures trends with momentum.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA(50) for trend filter ===
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h Williams Alligator ===
    # Jaw: SMA(13, 8) - slowest
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8, 5) - middle
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5, 3) - fastest
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bullish Alligator alignment: Lips > Teeth > Jaw (awakening uptrend)
        # 2. Price above 1d EMA50 (bullish trend bias)
        # 3. Volume confirmation
        if (lips[i] > teeth[i] and teeth[i] > jaw[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bearish Alligator alignment: Lips < Teeth < Jaw (awakening downtrend)
        # 2. Price below 1d EMA50 (bearish trend bias)
        # 3. Volume confirmation
        elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsAlligator_EMA50_VolFilter_v1"
timeframe = "12h"
leverage = 1.0