#!/usr/bin/env python3
"""
Experiment #1372: 12h Camarilla Pivot + Volume Spike + Chop Regime Filter
HYPOTHESIS: Camarilla pivot levels (H3/L3) from daily timeframe act as institutional support/resistance on 12h charts. 
Volume confirmation (>2.0x average) filters for significant participation. Choppiness index regime filter (CHOP > 61.8) ensures we only trade in ranging markets where mean reversion at pivots works. 
In bear markets (2025+), price tends to revert to mean at these levels rather than break out, making this a fading strategy with low trade frequency. 
ATR-based stoploss manages risk. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1372_12h_camarilla_vol_chop_v1"
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
    
    # Calculate Camarilla levels: H3/L3 = close ± 1.1*(high-low)/2
    # These are the key levels for mean reversion
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d / 2
    camarilla_l3 = close_1d - 1.1 * range_1d / 2
    
    # Align to 12h timeframe (with shift(1) for completed bars only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR1)/ATR(n)) / log10(n)
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market (avoid for mean reversion)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr14 = np.zeros(n)
    for i in range(14, n):
        sum_atr14[i] = np.sum(atr[i-13:i+1])
    
    chop = np.full(n, 50.0)  # default neutral
    for i in range(14, n):
        if atr[i] > 0 and sum_atr14[i] > 0:
            chop[i] = 100 * np.log10(sum_atr14[i] / atr[i]) / np.log10(14)
        else:
            chop[i] = 50.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for volume MA and CHOP
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss or mean reversion target ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                # Take profit: at Camarilla H3 level (mean reversion target)
                target_level = camarilla_h3_aligned[i]
                if low[i] < stop_level or high[i] >= target_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                # Take profit: at Camarilla L3 level (mean reversion target)
                target_level = camarilla_l3_aligned[i]
                if high[i] > stop_level or low[i] <= target_level:
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
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        if volume_spike and chop_filter:
            # Fade extreme moves: short near H3, long near L3
            # Only enter if price is near the Camarilla levels (within 0.5*ATR)
            near_h3 = abs(price - camarilla_h3_aligned[i]) < 0.5 * atr[i]
            near_l3 = abs(price - camarilla_l3_aligned[i]) < 0.5 * atr[i]
            
            if near_h3:
                # Short at H3 resistance (expect reversion to mean)
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            elif near_l3:
                # Long at L3 support (expect reversion to mean)
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