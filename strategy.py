#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Camarilla R1 AND 1d volume > 2.0x 20-period average AND choppiness < 61.8 (trending).
# Short when price breaks below Camarilla S1 AND 1d volume > 2.0x 20-period average AND choppiness < 61.8 (trending).
# Exit when price crosses Camarilla Pivot point (PP) OR volume drops below average OR choppiness > 61.8 (range).
# Uses discrete position size 0.25. Designed to capture intraday momentum in trending markets while avoiding chop.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag and maximize edge.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (based on prior 1d candles) ===
    # We'll use 1d data to calculate Camarilla levels for the 12h timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla: PP = (H+L+C)/3, Range = H-L
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    PP = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    R1 = close_1d + (range_1d * 1.1 / 12.0)
    S1 = close_1d - (range_1d * 1.1 / 12.0)
    
    # Align 1d Camarilla levels to 12h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # === 1d Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === 1d Indicators: Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    # We'll use a simplified version: CHOP = 100 * ATR(14) / (HHV(20) - LLV(20))
    # Where CHOP > 61.8 = range, CHOP < 38.2 = trending
    atr_14 = pd.Series(high).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(np.diff(x, prepend=x[0]))), raw=True
    ).values
    # Simplified ATR calculation for performance
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])), 
                                          np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    hh_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    ll_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    chop_denom = hh_20 - ll_20
    chop_denom = np.where(chop_denom == 0, 1, chop_denom)  # avoid division by zero
    chop = 100 * (atr_14 / chop_denom)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for MA, 14 for ATR)
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Camarilla PP OR volume spike ends OR chop > 61.8 (range)
            if price < PP_aligned[i] or not vol_spike or chop_val > 61.8:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Camarilla PP OR volume spike ends OR chop > 61.8 (range)
            if price > PP_aligned[i] or not vol_spike or chop_val > 61.8:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND chop < 61.8 (trending)
            if price > R1_aligned[i] and vol_spike and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND chop < 61.8 (trending)
            elif price < S1_aligned[i] and vol_spike and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dVolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0