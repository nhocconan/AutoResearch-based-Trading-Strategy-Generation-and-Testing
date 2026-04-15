#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d trend filter and ATR-based stop.
# Bull Power = High - EMA13, Bear Power = Low - EMA13. Uses 1d EMA50 for trend bias to avoid counter-trend trades.
# Volume filter (current volume > 1.3x 20-bar SMA) confirms momentum. Designed for low trade frequency (12-25/year)
# to minimize fee drag. Works in bull/bear: Elder Ray shows market power, 1d EMA filters direction, volume confirms.

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
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current 6h volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (bulls in control)
        # 2. Bull Power > Bear Power (bulls stronger than bears)
        # 3. 1d price above EMA50 (bullish long-term trend bias)
        # 4. Volume confirmation
        if (bull_power[i] > 0 and
            bull_power[i] > bear_power[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (bears in control)
        # 2. Bear Power < Bull Power (bears stronger than bulls)
        # 3. 1d price below EMA50 (bearish long-term trend bias)
        # 4. Volume confirmation
        elif (bear_power[i] < 0 and
              bear_power[i] < bull_power[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Elder_Ray_EMA50_VolFilter_v1"
timeframe = "6h"
leverage = 1.0