#!/usr/bin/env python3
"""
6h_TRIX_VolumeSpike_Regime
Hypothesis: TRIX (triple-smoothed EMA) detects momentum shifts in 6b candles, while volume spikes confirm breakout strength and Choppy Market Index (CMI) filters out ranging markets. Works in bull/bear by adapting to regime: trend-follow when CMI < 38.2 (trending), mean-revert when CMI > 61.8 (range). Target: 20-50 trades/year.
"""
name = "6h_TRIX_VolumeSpike_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for CMI regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX on 6b close (triple EMA of 1-period ROC)
    # TRIX = EMA(EMA(EMA(ROC, 12), 12), 12) * 100
    roc = np.diff(close, prepend=close[0]) / close  # 1-period ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 * 100
    
    # Calculate Choppiness Index (CMI) on 1d data
    # CMI = 100 * log10(sum(ATR, 14) / (HHH - LLL)) / log10(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll = df_1d['low'].rolling(window=14, min_periods=14).min().values
    cmi = 100 * np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum().values / (hh - ll)) / np.log10(14)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_avg * 2.0)
    
    # Align 1d CMI to 6b timeframe
    cmi_aligned = align_htf_to_ltf(prices, df_1d, cmi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for TRIX and volume
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(trix[i]) or np.isnan(cmi_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime filter: CMI < 38.2 = trending (trend follow), CMI > 61.8 = ranging (mean revert)
            if cmi_aligned[i] < 38.2:  # Trending regime
                # Long: TRIX rising + volume spike
                if trix[i] > trix[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX falling + volume spike
                elif trix[i] < trix[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            elif cmi_aligned[i] > 61.8:  # Ranging regime
                # Long: TRIX trough (reversal from negative) + volume spike
                if trix[i] > trix[i-1] and trix[i-1] < 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX peak (reversal from positive) + volume spike
                elif trix[i] < trix[i-1] and trix[i-1] > 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position != 0:
            # Exit: TRIX crosses zero or opposite regime signal
            if position == 1:
                if trix[i] < 0:  # TRIX turned negative
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if trix[i] > 0:  # TRIX turned positive
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals