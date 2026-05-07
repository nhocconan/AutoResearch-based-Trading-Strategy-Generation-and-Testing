#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_Regime
Hypothesis: TRIX (momentum) with volume spike and Choppiness regime filter on 12h.
Long when TRIX crosses above zero with volume spike in trending market (CHOP < 38.2).
Short when TRIX crosses below zero with volume spike in trending market.
Avoids choppy markets (CHOP > 61.8) to reduce false signals. Target: 15-25 trades/year.
"""
name = "12h_TRIX_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TRIX on 12h close: EMA(EMA(EMA(close, 12), 12), 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    
    # Calculate Choppiness Index on 1d: CHOP = 100 * LOG10(SUM(ATR,14)/ (MAX(HIGH,14)-MIN(LOW,14))) / LOG10(14)
    atr_list = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        else:
            tr = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
        atr_list.append(tr)
    
    atr = np.array(atr_list)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    range_14 = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / range_14) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 20)  # TRIX needs 3*12=36, vol needs 20
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_avg.iloc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + volume spike + trending market (CHOP < 38.2)
            if trix[i] > 0 and trix[i-1] <= 0 and volume_filter.iloc[i] and chop_aligned[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume spike + trending market (CHOP < 38.2)
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_filter.iloc[i] and chop_aligned[i] < 38.2:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: TRIX crosses back through zero
            if position == 1:
                if trix[i] < 0 and trix[i-1] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if trix[i] > 0 and trix[i-1] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals