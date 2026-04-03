#!/usr/bin/env python3
"""
Experiment #262: 12h Camarilla Pivot + Volume Spike + Chop Regime Strategy

HYPOTHESIS: Camarilla pivot levels from 1d combined with volume confirmation (>2.0x average) 
and choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend) captures high-probability 
breakouts in both bull and bear markets. In trending regimes (CHOP < 38.2), we trade breakouts 
from Camarilla H3/L3 levels with volume spike. In ranging markets (CHOP > 61.8), we avoid 
false breakouts. Uses ATR-based stoploss (2.0x) and minimum 6-bar holding period to reduce 
churn. Target: 75-150 total trades over 4 years on 12h timeframe.
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
    
    # === HTF: 1d data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla: H4 = C + (H-L)*1.1/2, H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4, L4 = C - (H-L)*1.1/2
    # where C = (H+L+C)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_values = typical_price.values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_h3 = typical_price_values + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = typical_price_values - (high_1d - low_1d) * 1.1 / 4
    camarilla_h4 = typical_price_values + (high_1d - low_1d) * 1.1 / 2
    camarilla_l4 = typical_price_values - (high_1d - low_1d) * 1.1 / 2
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr_12h = np.zeros(n)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime filter ===
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Sum of TR over period
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        # Highest high and lowest low over period
        hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        # Choppiness Index: 100 * log10(tr_sum / (hh - ll)) / log10(period)
        # Avoid division by zero
        hh_ll = hh - ll
        chop = np.full(len(high), 50.0)  # Default to neutral
        mask = (hh_ll > 0) & ~np.isnan(tr_sum) & ~np.isnan(hh_ll)
        chop[mask] = 100 * np.log10(tr_sum[mask] / hh_ll[mask]) / np.log10(period)
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
    
    warmup = 100  # Warmup for CHOP and volume MA stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # --- Chop Regime Filter: Only trade when CHOP < 38.2 (trending) ---
        is_trending_regime = chop[i] < 38.2
        is_ranging_regime = chop[i] > 61.8  # Strong ranging
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Breakout Conditions ---
        breakout_h3 = high[i] > camarilla_h3_aligned[i-1]  # Break above H3
        breakout_l3 = low[i] < camarilla_l3_aligned[i-1]   # Break below L3
        breakout_h4 = high[i] > camarilla_h4_aligned[i-1]  # Break above H4 (stronger)
        breakout_l4 = low[i] < camarilla_l4_aligned[i-1]   # Break below L4 (stronger)
        
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
                # Exit on opposite Camarilla breakout (contrarian exit)
                if breakout_l3 and volume_spike:
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
                # Exit on opposite Camarilla breakout (contrarian exit)
                if breakout_h3 and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 6 bars to reduce churn
            if bars_since_entry < 6:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade in trending regimes (CHOP < 38.2) with volume spike
        if is_trending_regime and volume_spike:
            # Long: Camarilla H3 breakout up
            if breakout_h3:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Camarilla L3 breakout down
            elif breakout_l3:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # In ranging regime or no volume spike, do not trade
            signals[i] = 0.0
    
    return signals