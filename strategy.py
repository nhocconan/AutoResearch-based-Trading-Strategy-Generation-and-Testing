#!/usr/bin/env python3
"""
Experiment #4211: 6h Donchian(20) breakout + 1d pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture momentum when aligned with 1d Camarilla pivot bias (price between H3/L3 for continuation, breaks of H4/L4 for acceleration) and confirmed by volume (>1.5x average). The 1d pivot provides institutional reference levels that work in both bull/bear markets by defining value areas and breakout zones. Discrete position sizing (0.25) limits fee churn, targeting 75-200 total trades over 4 years (19-50/year). Uses ATR-based trailing stop (2.5x) for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4211_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Precompute HTF: 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate Camarilla pivots from previous 1d bar
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        c_1d = df_1d['close'].values
        
        # True range for Camarilla calculation
        tr_1d = np.maximum(h_1d - l_1d, np.maximum(np.abs(h_1d - np.concatenate([[np.nan], c_1d[:-1]])), np.abs(l_1d - np.concatenate([[np.nan], c_1d[:-1]))))
        atr_1d = pd.Series(tr_1d).ewm(span=5, min_periods=1, adjust=False).mean().values  # ATR(5) for volatility
        
        # Camarilla levels: H4, H3, L3, L4
        pivot = (h_1d + l_1d + c_1d) / 3
        range_1d = h_1d - l_1d
        h4 = pivot + (range_1d * 1.1 / 2)
        h3 = pivot + (range_1d * 1.1 / 4)
        l3 = pivot - (range_1d * 1.1 / 4)
        l4 = pivot - (range_1d * 1.1 / 2)
        
        # Align to 6h timeframe (shift(1) already applied inside for no look-ahead)
        h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
        h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
        l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
        l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    else:
        h4_aligned = np.full(n, np.nan)
        h3_aligned = np.full(n, np.nan)
        l3_aligned = np.full(n, np.nan)
        l4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Pivot bias conditions
            price_above_h3 = price > h3_aligned[i]
            price_below_l3 = price < l3_aligned[i]
            price_between_h3_l3 = (price > l3_aligned[i]) & (price < h3_aligned[i])
            
            # Long conditions: Donchian breakout up + price above H3 (continuation/acceleration)
            long_entry = breakout_up and price_above_h3
            
            # Short conditions: Donchian breakout down + price below L3 (continuation/acceleration)
            short_entry = breakout_dn and price_below_l3
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals