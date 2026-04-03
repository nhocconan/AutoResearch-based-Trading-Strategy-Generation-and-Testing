#!/usr/bin/env python3
"""
Experiment #262: 12h Camarilla pivot + volume spike + chop regime filter
HYPOTHESIS: Camarilla pivot levels (H3/L3) from daily chart act as intraday support/resistance on 12h. 
Long when price touches L3 with volume spike (>2.0x) in choppy market (CHOP>61.8). 
Short when price touches H3 with volume spike in choppy market. 
Chop regime filter ensures mean-reversion behavior at pivots. 
ATR stoploss at 2.5x. Discrete sizing 0.25 minimizes fee drag. 
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_262_12h_camarilla_pivot_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior day's OHLC
    # H3/L3 are the key levels for intraday reversals
    prior_high = df_1d['high'].shift(1)
    prior_low = df_1d['low'].shift(1)
    prior_close = df_1d['close'].shift(1)
    
    pivot = (prior_high + prior_low + prior_close) / 3.0
    range_ = prior_high - prior_low
    
    h3 = prior_close + range_ * 1.1 / 4
    l3 = prior_close - range_ * 1.1 / 4
    h4 = prior_close + range_ * 1.1 / 2
    l4 = prior_close - range_ * 1.1 / 2
    
    # Align to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4.values)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4.values)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime filter ===
    # CHOP > 61.8 = ranging market (good for mean reversion at pivots)
    # CHOP < 38.2 = trending market (avoid)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    high_low_range = pd.Series(high).rolling(window=14, min_periods=14).max() - pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = np.zeros(n)
    chop[13:] = 100 * np.log10(atr_sum[13:] / high_low_range[13:]) / np.log10(14)
    chop[:13] = 50.0  # neutral default
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 30  # Enough for 20-period volume MA and 14-period indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filter: Only trade in choppy/ranging markets (CHOP > 61.8) ---
        chop_regime = chop[i] > 61.8
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Proximity to Pivot Levels (within 0.5% of H3/L3) ---
        near_h3 = abs(price - h3_aligned[i]) / h3_aligned[i] < 0.005
        near_l3 = abs(price - l3_aligned[i]) / l3_aligned[i] < 0.005
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require chop regime + volume spike + pivot proximity
        if chop_regime and volume_spike:
            # Long: price near L3 (support) with volume spike
            if near_l3:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price near H3 (resistance) with volume spike
            elif near_h3:
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