#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels (R1/S1) with volume spike and choppiness regime filter.
# Long when price breaks above R1 with volume spike in trending regime (CHOP < 38.2).
# Short when price breaks below S1 with volume spike in trending regime.
# Uses tight entry conditions to limit trades (target: 20-50/year).
# Works in both bull and bear markets via regime filter that adapts to market conditions.
# Position size: 0.25.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Choppiness Index (14-period) ===
    atr_list = []
    for i in range(n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]) if i > 0 else 0,
            abs(low[i] - close[i-1]) if i > 0 else 0
        )
        atr_list.append(tr)
    
    atr_14 = pd.Series(atr_list).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(atr_list).rolling(window=14, min_periods=14).sum().values
    max_hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(sum_tr_14 / (max_hh_14 - min_ll_14)) / np.log10(14)
    chop = np.where((max_hh_14 - min_ll_14) == 0, 100, chop)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Typical price
    tp_1d = (close_1d + high_1d + low_1d) / 3
    
    # Camarilla levels
    r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align to 4h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d Volume Spike ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(chop[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        
        # Regime filter: trending only (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        # EXIT LOGIC
        exit_signal = False
        if position == 1:  # Long
            # Exit if price breaks below S1
            if price < s1_1d_aligned[i]:
                exit_signal = True
        elif position == -1:  # Short
            # Exit if price breaks above R1
            if price > r1_1d_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # ENTRY LOGIC (only when flat)
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + trending regime
            if price > r1_1d_aligned[i] and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + trending regime
            elif price < s1_1d_aligned[i] and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1S1_VolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0