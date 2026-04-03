#!/usr/bin/env python3
"""
Experiment #1065: 12h Camarilla Pivot + Volume Spike + Chop Regime Filter
HYPOTHESIS: Camarilla pivot levels from 1d act as intraday support/resistance. Long when price touches L3 with volume spike and choppy market (CHOP>61.8). Short when price touches H3 with volume spike and choppy market. Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1065_12h_camarilla_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d: based on previous day's range
    # L3 = close - (high - low) * 1.1/4
    # H3 = close + (high - low) * 1.1/4
    range_1d = high_1d - low_1d
    camarilla_l3 = close_1d - range_1d * 1.1 / 4
    camarilla_h3 = close_1d + range_1d * 1.1 / 4
    
    # Align to 12h timeframe (shifted by 1 for completed bars only)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: Choppiness Index (CHOP) regime filter ===
    # CHOP = 100 * log10(sum(ATR) / (max(high)-min(low))) / log10(n)
    # CHOP > 61.8 = ranging market (good for mean reversion at pivots)
    # CHOP < 38.2 = trending market
    atr = np.zeros(n)
    for i in range(1, n):
        atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr[0] = high[0] - low[0]
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.full(n, np.nan)
    denominator = max_high - min_low
    chop[14:] = 100 * np.log10(atr_sum[14:] / denominator[14:]) / np.log10(14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for volume MA and chop
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: time-based exit after 4 bars (~2d on 12h) to avoid overtrading ---
        if in_position:
            bars_since_entry += 1
            
            if bars_since_entry > 4:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        # Regime filter: choppy market (CHOP > 61.8) for mean reversion at pivots
        choppy_market = chop[i] > 61.8
        
        if volume_spike and choppy_market:
            # Touch L3 (support) with volume spike in choppy market -> long
            if low[i] <= camarilla_l3_aligned[i] * 1.001:  # small tolerance for touch
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Touch H3 (resistance) with volume spike in choppy market -> short
            elif high[i] >= camarilla_h3_aligned[i] * 0.999:  # small tolerance for touch
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

if __name__ == "__main__":
    # For testing only - remove in production
    pass