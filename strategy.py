#!/usr/bin/env python3
"""
Experiment #025: 12h Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels from 1d combined with volume confirmation (>1.5x average) 
and choppiness regime filter (CHOP > 61.8 for mean reversion) captures high-probability 
reversals at key support/resistance levels. In choppy markets (range-bound), we trade 
mean reversion from Camarilla H3/L3 levels with volume confirmation. Uses ATR-based 
stoploss (2.0x) and discrete position sizing (0.25) to minimize fee churn. 
Target: 75-150 trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_025_12h_camarilla_pivot_vol_chop_v1"
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
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: Based on previous day's high, low, close
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.125 * (High - Low)
    # L3 = Close - 1.125 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # We'll use H3/L3 for mean reversion entries
    
    # Shift by 1 to use previous day's data (no look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels using previous day's data
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.125 * range_1d
    camarilla_l3 = close_1d - 1.125 * range_1d
    
    # Align to 12h timeframe (with shift(1) inside align_htf_to_ltf)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    def calculate_tr(high, low, close_prev):
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close_prev[i-1]), abs(low[i] - close_prev[i-1]))
        return tr
    
    tr_12h = calculate_tr(high, low, close)
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime detection ===
    def calculate_chop(high, low, close, period=14):
        """Choppiness Index: Higher values indicate ranging market, lower values indicate trending"""
        atr_sum = pd.Series(calculate_tr(high, low, close)).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        # Avoid division by zero
        range_max_min = highest_high - lowest_low
        chop = np.zeros(n)
        for i in range(n):
            if range_max_min[i] > 0 and not np.isnan(atr_sum[i]):
                chop[i] = 100 * np.log10(atr_sum[i] / range_max_min[i]) / np.log10(period)
            else:
                chop[i] = 50.0  # Neutral when undefined
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Sufficient warmup for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Chop > 61.8 = ranging market (good for mean reversion) ---
        is_ranging = chop[i] > 61.8
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        price = close[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Mean reversion exit: price reaches opposite Camarilla level
            if position_side > 0:  # Long - exit at H3
                if price >= h3_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short - exit at L3
                if price <= l3_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat and Ranging) ---
        # Only trade in ranging markets (mean reversion)
        if is_ranging:
            # Long: Price touches/below L3 with volume spike
            if price <= l3_level and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Price touches/above H3 with volume spike
            elif price >= h3_level and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # In trending markets, do not trade (avoid false mean reversion signals)
            signals[i] = 0.0
    
    return signals