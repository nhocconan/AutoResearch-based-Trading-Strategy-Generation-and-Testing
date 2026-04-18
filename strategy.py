#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime
Hypothesis: Price breaks above/below daily Camarilla R1/S1 with volume spike and volatility regime filter (Choppiness Index > 61.8 = range-bound = mean reversion opportunity). Works in bull/bear by capturing reversals from extreme levels in ranging markets. Target: 25-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla and Choppiness
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0  # pivot point
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Choppiness Index (14) on daily: >61.8 = ranging (good for mean reversion)
    atr_1d = []
    for i in range(len(high_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(atr_ma_1d / (np.sum(atr_1d[-14:]) if len(atr_1d) >= 14 else np.sum(atr_1d))) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # Volume spike: >2.0x 20-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pp_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # Only trade in ranging markets (Choppiness > 61.8)
        if chop_val <= 61.8:
            # In trending markets, stay flat to avoid whipsaw
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks below S1 (oversold) with volume spike -> mean reversion long
            if price < s1_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks above R1 (overbought) with volume spike -> mean reversion short
            elif price > r1_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to pivot point (mean reversion complete)
            if price >= pp_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to pivot point
            if price <= pp_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0