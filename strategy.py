#!/usr/bin/env python3
"""
Experiment #114: 1h Camarilla pivot + volume spike + choppiness regime filter
HYPOTHESIS: Camarilla pivot levels on 4h provide institutional support/resistance. 
Volume spikes confirm participation. Choppiness index regime filter avoids whipsaws 
in sideways markets. Uses 4h/1d for signal direction, 1h only for entry timing precision. 
Session filter (08-20 UTC) reduces noise trades. Targets 15-37 trades/year on 1h timeframe 
(60-150 total over 4 years) to minimize fee drag while maintaining edge in both bull 
and bear markets through mean reversion at extremes and trend continuation in momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_camarilla_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 4h data for Camarilla pivots (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels on 4h
    if len(df_4h) >= 2:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        close_4h = df_4h['close'].values
        
        camarilla_h3 = np.zeros(len(close_4h))
        camarilla_l3 = np.zeros(len(close_4h))
        camarilla_h4 = np.zeros(len(close_4h))
        camarilla_l4 = np.zeros(len(close_4h))
        
        for i in range(len(close_4h)):
            if i >= 1:  # Need previous bar
                range_4h = high_4h[i-1] - low_4h[i-1]
                camarilla_h3[i] = close_4h[i-1] + range_4h * 1.1 / 6
                camarilla_l3[i] = close_4h[i-1] - range_4h * 1.1 / 6
                camarilla_h4[i] = close_4h[i-1] + range_4h * 1.1 / 2
                camarilla_l4[i] = close_4h[i-1] - range_4h * 1.1 / 2
            else:
                camarilla_h3[i] = close_4h[i]
                camarilla_l3[i] = close_4h[i]
                camarilla_h4[i] = close_4h[i]
                camarilla_l4[i] = close_4h[i]
        
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    else:
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for choppiness regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        chop = np.zeros(len(close_1d))
        atr_14 = np.zeros(len(close_1d))
        
        for i in range(len(close_1d)):
            if i == 0:
                tr = high_1d[i] - low_1d[i]
            else:
                tr = max(high_1d[i] - low_1d[i], 
                         abs(high_1d[i] - close_1d[i-1]), 
                         abs(low_1d[i] - close_1d[i-1]))
                atr_14[i] = (atr_14[i-1] * 13 + tr) / 14 if i >= 1 else tr
            
            if i >= 13:
                atr_sum = np.sum(atr_14[i-13:i+1])
                hh = np.max(high_1d[i-13:i+1])
                ll = np.min(low_1d[i-13:i+1])
                if hh - ll > 0:
                    chop[i] = 100 * np.log10(atr_sum / np.log2(14) / (hh - ll))
                else:
                    chop[i] = 50.0
            else:
                chop[i] = 50.0
        
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio on 1w
    if len(df_1w) >= 10:
        vol_1w = df_1w['volume'].values
        vol_ma_10 = np.zeros(len(vol_1w))
        for i in range(len(vol_1w)):
            if i >= 9:
                vol_ma_10[i] = np.mean(vol_1w[i-9:i+1])
            else:
                vol_ma_10[i] = np.mean(vol_1w[0:i+1]) if i >= 0 else vol_1w[i]
        
        vol_ratio_1w = np.ones(len(vol_1w))
        vol_ratio_1w[10:] = vol_1w[10:] / vol_ma_10[10:]
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === 1h Indicators ===
    # ATR(14) for dynamic thresholds
    atr_1h = np.zeros(n)
    tr_1h = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr_1h[i] = high[i] - low[i]
        else:
            tr_1h[i] = max(high[i] - low[i], 
                           abs(high[i] - close[i-1]), 
                           abs(low[i] - close[i-1]))
        if i == 0:
            atr_1h[i] = tr_1h[i]
        else:
            atr_1h[i] = (atr_1h[i-1] * 13 + tr_1h[i]) / 14
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Chop > 61.8 = range (mean revert), Chop < 38.2 = trending ---
        is_range = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) on 1w ---
        volume_spike = vol_ratio_1w_aligned[i] > 1.5
        
        # --- Mean Reversion Logic in Range ---
        if is_range and volume_spike:
            # Long near L3/L4 support
            if close[i] <= camarilla_l3_aligned[i] * 1.001:  # Small buffer for entry
                signals[i] = SIZE
            # Short near H3/H4 resistance
            elif close[i] >= camarilla_h3_aligned[i] * 0.999:
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        
        # --- Breakout Logic in Trending ---
        elif is_trending and volume_spike:
            # Long breakout above H4
            if close[i] > camarilla_h4_aligned[i]:
                signals[i] = SIZE
            # Short breakdown below L4
            elif close[i] < camarilla_l4_aligned[i]:
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        
        else:
            signals[i] = 0.0
    
    return signals