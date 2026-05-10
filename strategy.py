#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_Regime - TRIX momentum with volume spike and Chop regime filter
# Hypothesis: TRIX (Triple Exponential Average) captures momentum shifts. In trending markets,
# TRIX crossovers signal strong moves. Combined with volume spikes (institutional interest)
# and Chop regime filter (avoid ranging markets), this should work in both bull and bear
# by only taking long signals when TRIX > 0 and short when TRIX < 0.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for regime filter (Chop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate TRIX (15,9,9) - standard settings
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = (ema3.pct_change() * 100).values  # TRIX as percentage
    
    # Calculate Chop index (14-period) for regime filter
    # Chop = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)  # Avoid division by zero
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX (15+9+9=33), Chop (14), volume MA (20)
    start_idx = max(33, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: Chop > 61.8 = ranging (avoid), Chop < 38.2 = trending
        trending_regime = chop[i] < 38.2
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: TRIX > 0 (bullish momentum) + trending regime + volume spike
            if trix[i] > 0 and trending_regime and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX < 0 (bearish momentum) + trending regime + volume spike
            elif trix[i] < 0 and trending_regime and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative or regime changes to ranging
            if trix[i] <= 0 or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive or regime changes to ranging
            if trix[i] >= 0 or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals