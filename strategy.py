#!/usr/bin/env python3
"""
Experiment #322: 12h Camarilla Pivot Breakout + 1d Volume Spike + Chop Regime Filter

HYPOTHESIS: 12h Camarilla pivot levels (L3, L4, H3, H4) act as significant support/resistance.
Breakouts above H3/H4 or below L3/L4 with volume confirmation (>2.0x average) and 
favorable chop regime (CHOP > 61.8 for mean reversion setups, CHOP < 38.2 for trend continuation)
capture high-probability moves. The 12h timeframe targets 12-37 trades/year (50-150 total) 
to minimize fee drag. Works in bull markets (breakouts with volume) and bear markets 
(mean reversion from extreme levels in choppy conditions). Uses ATR-based stoploss for risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_322_12h_camarilla_1d_volume_chop_v1"
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
    
    # Calculate Camarilla pivot levels for 1d
    def calculate_camarilla(h, l, c):
        # Camarilla levels based on previous day's range
        range_ = h - l
        if range_ <= 0:
            return np.full_like(c, np.nan), np.full_like(c, np.nan), np.full_like(c, np.nan), np.full_like(c, np.nan)
        
        # Camarilla levels: H4, H3, L3, L4
        h4 = c + range_ * 1.1 / 2
        h3 = c + range_ * 1.1 / 4
        l3 = c - range_ * 1.1 / 4
        l4 = c - range_ * 1.1 / 2
        
        return h4, h3, l3, l4
    
    # Shift 1d data by 1 to avoid look-ahead (use previous day's levels)
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    h4_1d, h3_1d, l3_1d, l4_1d = calculate_camarilla(h_1d, l_1d, c_1d)
    
    # Align to 12h timeframe with proper shift(1) for completed bars only
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
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
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime filter ===
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First period
        
        # Sum of TR over period
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        # Highest high and lowest low over period
        hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        # Choppiness Index
        chop = np.zeros_like(close)
        denominator = hh - ll
        # Avoid division by zero
        mask = (denominator != 0) & (~np.isnan(tr_sum)) & (~np.isnan(denominator))
        chop[mask] = 100 * np.log10(tr_sum[mask] / denominator[mask]) / np.log10(period)
        chop[~mask] = 50.0  # Neutral when invalid
        
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
    
    warmup = 100  # Warmup for Camarilla (needs 1d data) and CHOP(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Chop Regime Filter ---
        # CHOP > 61.8 = ranging market (favor mean reversion)
        # CHOP < 38.2 = trending market (favor breakout continuation)
        chop_high = chop[i] > 61.8  # Ranging regime
        chop_low = chop[i] < 38.2   # Trending regime
        
        # --- Camarilla Breakout Conditions ---
        breakout_h3 = close[i] > h3_1d_aligned[i]  # Break above H3
        breakdown_l3 = close[i] < l3_1d_aligned[i]  # Break below L3
        breakout_h4 = close[i] > h4_1d_aligned[i]  # Break above H4 (stronger)
        breakdown_l4 = close[i] < l4_1d_aligned[i]  # Break below L4 (stronger)
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if close[i] < l3_1d_aligned[i]:  # Long TP at L3
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
                # Take profit at opposite Camarilla level
                if close[i] > h3_1d_aligned[i]:  # Short TP at H3
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Regime-adaptive entry logic:
        # In trending market (CHOP < 38.2): breakout continuation
        # In ranging market (CHOP > 61.8): mean reversion from extreme levels
        
        # Long conditions:
        # 1. Trending market: H3/H4 breakout with volume
        # 2. Ranging market: Oversold bounce from L4/L3 with volume
        long_trending = chop_low and (breakout_h3 or breakout_h4) and volume_spike
        long_ranging = chop_high and (close[i] < l4_1d_aligned[i]) and volume_spike and (close[i] > low[i])  # Bounce from low
        
        # Short conditions:
        # 1. Trending market: L3/L4 breakdown with volume
        # 2. Ranging market: Overbought rejection from H4/H3 with volume
        short_trending = chop_low and (breakdown_l3 or breakdown_l4) and volume_spike
        short_ranging = chop_high and (close[i] > h4_1d_aligned[i]) and volume_spike and (close[i] < high[i])  # Rejection from high
        
        if long_trending or long_ranging:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_trending or short_ranging:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals