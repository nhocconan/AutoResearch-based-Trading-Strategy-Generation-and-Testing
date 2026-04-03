#!/usr/bin/env python3
"""
Experiment #1165: 12h Camarilla Pivot + 1d Volume Spike + Choppiness Regime
HYPOTHESIS: Camarilla pivot levels from 1d timeframe provide high-probability reversal zones. 
Volume confirmation (>2.0x average) ensures institutional participation. 
Choppiness filter (CHOP > 61.8) ensures we only trade in ranging markets where mean reversion works.
Designed to work in both bull (mean reversion in ranges) and bear (mean reversion in ranges) markets. 
Target: 75-150 total trades over 4 years (19-37/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1165_12h_camarilla_pivot_1d_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels and choppiness (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Camarilla pivot levels from previous 1d bar
    # H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    # H3 = Close + 1.1*(High-Low)/4, L3 = Close - 1.1*(High-Low)/4
    # H2 = Close + 1.1*(High-Low)/6, L2 = Close - 1.1*(High-Low)/6
    # H1 = Close + 1.1*(High-Low)/12, L1 = Close - 1.1*(High-Low)/12
    rng_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * rng_1d / 2
    camarilla_l4 = close_1d - 1.1 * rng_1d / 2
    camarilla_h3 = close_1d + 1.1 * rng_1d / 4
    camarilla_l3 = close_1d - 1.1 * rng_1d / 4
    camarilla_h2 = close_1d + 1.1 * rng_1d / 6
    camarilla_l2 = close_1d - 1.1 * rng_1d / 6
    camarilla_h1 = close_1d + 1.1 * rng_1d / 12
    camarilla_l1 = close_1d - 1.1 * rng_1d / 12
    
    # Align Camarilla levels to 12h timeframe (use previous 1d bar's levels)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Choppiness Index on 1d timeframe
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    # Simplified: use ATR(14) and normalize
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    sum_atr_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_atr_1d / (14 * np.log10(14))) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for volume MA and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h2_aligned[i]) or np.isnan(l2_aligned[i]) or
            np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        # Choppiness filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        if volume_spike and chop_filter:
            # Mean reversion at Camarilla levels
            # Short at resistance levels (H3, H4) when price reaches them
            # Long at support levels (L3, L4) when price reaches them
            if price >= h3_aligned[i]:  # Short at H3 or H4
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            elif price <= l3_aligned[i]:  # Long at L3 or L4
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

}