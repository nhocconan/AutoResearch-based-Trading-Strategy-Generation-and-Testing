#!/usr/bin/env python3
"""
Hypothesis: 4h TRIX crossover with 1d volume spike and chop regime filter.
- TRIX (12,9,9) crossover signals momentum shifts.
- Volume > 2.0x 20-bar average confirms institutional participation.
- Choppiness Index (14) > 61.8 = ranging market (mean reversion), < 38.2 = trending (momentum).
- In ranging markets, we fade TRIX crossovers; in trending markets, we follow them.
- Designed for 4h timeframe to capture medium-term swings in both bull and bear markets.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 30-60 trades/year (120-240 total over 4 years) to stay fee-efficient.
"""

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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume spike confirmation
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > 2.0 * vol_ma_1d
    
    # 1d Choppiness Index (14)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    sum_atr14 = atr14.rolling(window=14, min_periods=14).sum()
    max_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    min_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(sum_atr14 / (np.log10(14) * (max_high - min_low)))
    chop_values = chop.values
    
    # Align HTF indicators to LTF
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # TRIX (12,9,9) on LTF close
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean()
    trix_hist = trix - trix_signal
    
    # Align TRIX histogram (already LTF, but ensure proper alignment)
    trix_hist_values = trix_hist.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(trix_hist_values[i]) or np.isnan(trix_hist_values[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation and regime filter
        volume_confirm = vol_spike_1d_aligned[i] > 0.5  # boolean as float
        chop_value = chop_aligned[i]
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        # TRIX histogram crossover
        trix_cross_up = trix_hist_values[i-1] <= 0 and trix_hist_values[i] > 0
        trix_cross_down = trix_hist_values[i-1] >= 0 and trix_hist_values[i] < 0
        
        if position == 0:
            # In ranging market: mean reversion - fade TRIX crossovers
            if is_ranging and volume_confirm:
                if trix_cross_down:  # TRIX turning down -> short
                    signals[i] = -0.25
                    position = -1
                elif trix_cross_up:  # TRIX turning up -> long
                    signals[i] = 0.25
                    position = 1
            # In trending market: follow TRIX crossovers
            elif is_trending and volume_confirm:
                if trix_cross_up:  # TRIX turning up -> long
                    signals[i] = 0.25
                    position = 1
                elif trix_cross_down:  # TRIX turning down -> short
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: TRIX histogram crosses below zero OR chop exits trending/ranging
            if trix_cross_down or not (is_ranging or is_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX histogram crosses above zero OR chop exits trending/ranging
            if trix_cross_up or not (is_ranging or is_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0