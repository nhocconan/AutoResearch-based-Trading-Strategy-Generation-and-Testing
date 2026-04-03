#!/usr/bin/env python3
"""
Experiment #055: 6h Camarilla Pivot + Volume Spike + 1w Trend Filter
HYPOTHESIS: Price rejecting Camarilla R3/S3 levels with volume spike (>1.8x) and alignment to 1-week trend (close > 1w EMA20 for longs, < for shorts) captures institutional reaction at key levels. Weekly trend filter ensures we trade with higher timeframe momentum, reducing whipsaw in sideways markets. Discrete sizing (0.25) and ATR(14) stoploss (2.5*ATR). Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_055_6h_camarilla_pivot_vol_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior day
    # Camarilla: 
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    # We'll use H3/L3 for reversals and H4/L4 for breakouts
    hl_range = df_1d['high'] - df_1d['low']
    camarilla_h3 = df_1d['close'] + 1.1 * hl_range
    camarilla_l3 = df_1d['close'] - 1.1 * hl_range
    camarilla_h4 = df_1d['close'] + 1.5 * hl_range
    camarilla_l4 = df_1d['close'] - 1.5 * hl_range
    
    # Shift by 1 to use prior completed day's levels
    h3 = camarilla_h3.shift(1).values
    l3 = camarilla_l3.shift(1).values
    h4 = camarilla_h4.shift(1).values
    l4 = camarilla_l4.shift(1).values
    
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    ewma_20 = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ewma_20_aligned = align_htf_to_ltf(prices, df_1w, ewma_20)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ewma_20_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Trend Filter: 1-week EMA20 ---
        uptrend = price > ewma_20_aligned[i]
        downtrend = price < ewma_20_aligned[i]
        
        # --- Camarilla Reaction Logic ---
        # Reversal at H3/L3: price rejects these levels
        reject_h3 = (high[i] >= h3_aligned[i] and price < h3_aligned[i]) or \
                    (low[i] <= l3_aligned[i] and price > l3_aligned[i])
        reject_l3 = reject_h3  # same logic, symmetric
        
        # Breakout at H4/L4: price closes beyond these levels
        breakout_h4 = price > h4_aligned[i]
        breakout_l4 = price < l4_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~48h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: rejection at L3 (support hold) AND uptrend on 1w
            if reject_l3 and uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: rejection at H3 (resistance hold) AND downtrend on 1w
            elif reject_h3 and downtrend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            # Long breakout: break above H4 AND uptrend
            elif breakout_h4 and uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short breakout: break below L4 AND downtrend
            elif breakout_l4 and downtrend:
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