#!/usr/bin/env python3
"""
4h TRIX + Volume Spike + Choppiness Regime Filter
Hypothesis: TRIX (12-period) captures momentum with reduced lag. Long when TRIX crosses above zero with volume spike and non-choppy regime; short when crosses below zero. Volume confirms institutional participation, chop filter avoids whipsaws in ranging markets. Works in bull/bear via discrete sizing (0.25) and regime alignment. Uses 4h primary timeframe for signal generation.
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
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate TRIX: triple EMA of ROC, then ROC of that
    # ROC = (close / close.shift(1) - 1) * 100
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] / close[:-1] - 1) * 100
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # TRIX = ROC of triple EMA
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] / ema3[:-1] - 1) * 100
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # Choppiness regime filter: CHOP(14) > 61.8 = range (avoid), < 38.2 = trend (favor)
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        hhll = highest_high - lowest_low
        
        atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        chop = 100 * np.log10(atr_sum / np.log(10) / hhll)
        return chop
    
    chop_values = calculate_chop(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for TRIX warmup and volume MA
    start_idx = max(40, 21)  # TRIX needs ~36 (12*3), vol MA 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(chop_values[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade when NOT choppy (CHOP < 61.8 = trending)
        not_choppy = chop_values[i] < 61.8
        
        # TRIX signals: zero-cross with confirmation
        if i > 0:
            trix_prev = trix[i-1]
            trix_curr = trix[i]
            
            if position == 0:
                # Look for entry signals - require: TRIX zero-cross + volume + regime
                # Long: TRIX crosses above zero AND volume spike AND not choppy
                long_entry = (trix_prev <= 0) and (trix_curr > 0) and vol_spike and not_choppy
                # Short: TRIX crosses below zero AND volume spike AND not choppy
                short_entry = (trix_prev >= 0) and (trix_curr < 0) and vol_spike and not_choppy
                
                if long_entry:
                    signals[i] = 0.25
                    position = 1
                elif short_entry:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Long position management
                # Exit: TRIX crosses below zero OR loss of volume spike OR choppy regime
                if (trix_curr < 0) or (not vol_spike) or (chop_values[i] >= 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position management
                # Exit: TRIX crosses above zero OR loss of volume spike OR choppy regime
                if (trix_curr > 0) or (not vol_spike) or (chop_values[i] >= 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0