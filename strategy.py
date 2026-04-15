#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with volume confirmation and ATR-based stoploss
# In ranging markets (price within Donchian bands), fade extremes at bands; in trending markets (price outside bands),
# breakout continuation. Volume filter ensures momentum validity. Designed for low trade frequency 
# (20-50/year) to minimize fee drag while adapting to regime via Donchian structure.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian upper and lower bands
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # === 4h Indicators: ATR for volatility and stoploss ===
    # True Range components
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:14])  # seed with simple average
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Ranging market: price within Donchian bands
        # Trending market: price outside Donchian bands
        # Transition zone: at bands (no trade)
        
        in_range = (lower_aligned[i] < close[i] < upper_aligned[i])
        above_upper = close[i] >= upper_aligned[i]
        below_lower = close[i] <= lower_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. In ranging market AND price at lower band (mean reversion long)
        # 2. OR breakout above upper band (continuation long)
        # 3. Volume confirmation
        if vol_confirm:
            if (in_range and close[i] <= lower_aligned[i] * 1.001) or \
               (above_upper):
                signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In ranging market AND price at upper band (mean reversion short)
        # 2. OR breakdown below lower band (continuation short)
        # 3. Volume confirmation
        elif vol_confirm:
            if (in_range and close[i] >= upper_aligned[i] * 0.999) or \
               (below_lower):
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_MeanRev_Trend_v1"
timeframe = "4h"
leverage = 1.0