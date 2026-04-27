#!/usr/bin/env python3
"""
4h_TRIX_Volume_Spike_Regime
Hypothesis: TRIX (triple-smoothed EMA) captures momentum with less noise. Combining TRIX crossovers with volume spikes and a 1-day chop regime filter yields high-probability trades. The chop filter avoids whipsaws in ranging markets, while TRIX + volume confirms momentum bursts. Works in bull via TRIX>0 + volume, in bear via TRIX<0 + volume. Targets ~25 trades/year on 4h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period triple EMA, then 1-period percent change)
    # TRIX = [(EMA(EMA(EMA(close,15),15),15) - prev) / prev] * 100
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix_raw.fillna(0).values
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Choppy Index (high-low range vs ATR)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    range_1d = high_1d - low_1d
    
    # Chop = 100 * log10(sum(range_1d,14) / (atr_1d*14)) / log10(14)
    # Simplified: chop = 100 * (range_sum / (atr*14)) / log10(14) but we use common approximation
    range_sum = pd.Series(range_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * (range_sum / (atr_1d * 14)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (trend follow)
    # We want trending markets: chop < 38.2
    chop_trim = chop < 38.2
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_trim)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for TRIX and chop
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
        
        trix_now = trix[i]
        trix_prev = trix[i-1] if i > 0 else 0
        chop_now = chop_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike in trending market
            if trix_now > 0 and trix_prev <= 0 and vol_spike_val and chop_now:
                signals[i] = size
                position = 1
            # Short: TRIX crosses below zero with volume spike in trending market
            elif trix_now < 0 and trix_prev >= 0 and vol_spike_val and chop_now:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TRIX crosses below zero or volume dries up
            if trix_now < 0 or not vol_spike_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TRIX crosses above zero or volume dries up
            if trix_now > 0 or not vol_spike_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_Volume_Spike_Regime"
timeframe = "4h"
leverage = 1.0