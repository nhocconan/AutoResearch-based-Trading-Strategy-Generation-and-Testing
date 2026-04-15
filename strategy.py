#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA200 trend filter + Donchian(20) breakout + volume confirmation.
# In bull markets (price > 1w EMA200), go long on upper Donchian breakout; in bear markets (price < 1w EMA200),
# go short on lower Donchian breakout. Volume filter ensures breakout validity. Designed for low trade frequency
# (7-25/year) to minimize fee drag while adapting to trend via 1w EMA200.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: EMA(200) for trend filter ===
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    # Use rolling window with min_periods
    upper_donchian = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period volume SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_sma_20 * 1.5)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(upper_donchian[i]) or
            np.isnan(lower_donchian[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In bull trend (price > 1w EMA200)
        # 2. Price breaks above upper Donchian(20)
        # 3. Volume confirmation
        if (close[i] > ema_200_1w_aligned[i] and
            close[i] > upper_donchian[i] and
            vol_confirm[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In bear trend (price < 1w EMA200)
        # 2. Price breaks below lower Donchian(20)
        # 3. Volume confirmation
        elif (close[i] < ema_200_1w_aligned[i] and
              close[i] < lower_donchian[i] and
              vol_confirm[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_1wEMA200_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0