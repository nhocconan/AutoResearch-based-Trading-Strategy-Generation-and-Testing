#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter and volume spike confirmation.
# Williams %R(14) identifies overbought/oversold conditions. In 6h timeframe, extreme readings
# (> -20 for overbought, < -80 for oversold) combined with 1d EMA(50) trend filter provide
# high-probability mean reversion entries. Volume spike (> 2x 20-bar SMA) confirms momentum
# behind the reversal. Designed for low trade frequency (12-37/year) to minimize fee drag.
# Works in bull/bear: 1d EMA avoids counter-trend trades, Williams %R captures exhaustion.

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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicators: EMA(50) for trend filter ===
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    for i in range(warmup, n):
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_spike = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R indicates oversold (< -80)
        # 2. 6h price above 1d EMA50 (bullish trend bias)
        # 3. Volume spike confirms buying interest
        if (williams_r[i] < -80.0 and
            close[i] > ema_50_1d_aligned[i] and
            vol_spike):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R indicates overbought (> -20)
        # 2. 6h price below 1d EMA50 (bearish trend bias)
        # 3. Volume spike confirms selling interest
        elif (williams_r[i] > -20.0 and
              close[i] < ema_50_1d_aligned[i] and
              vol_spike):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_EMA50_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0