#!/usr/bin/env python3
"""
Experiment #233: 4h Camarilla Pivot + Volume Spike + Chop Regime (12h HTF)

HYPOTHESIS: Camarilla pivot levels from 12h timeframe act as strong support/resistance zones.
In trending markets (12h ADX > 25), we trade breakouts of H3/L3 levels with volume confirmation.
In ranging markets (12h ADX < 25), we mean-revert from H4/L4 levels with volume confirmation.
This adapts to both bull (strong breakouts) and bear (failed reversals, range-bound) markets.
4h timeframe targets 19-50 trades/year (75-200 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_233_4h_camarilla_pivot_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivots and regime detection (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h typical price for pivot calculation
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    pivot_12h = typical_price_12h.values
    # Range = H - L
    range_12h = (high_12h - low_12h).values
    
    # Camarilla levels:
    # H4 = Pivot + Range * 1.1/2
    # L4 = Pivot - Range * 1.1/2
    # H3 = Pivot + Range * 1.1/4
    # L3 = Pivot - Range * 1.1/4
    # H2 = Pivot + Range * 1.1/6
    # L2 = Pivot - Range * 1.1/6
    # H1 = Pivot + Range * 1.1/12
    # L1 = Pivot - Range * 1.1/12
    
    multiplier = 1.1 / 12.0
    h4_12h = pivot_12h + range_12h * multiplier * 6  # *1.1/2
    l4_12h = pivot_12h - range_12h * multiplier * 6  # *1.1/2
    h3_12h = pivot_12h + range_12h * multiplier * 3  # *1.1/4
    l3_12h = pivot_12h - range_12h * multiplier * 3  # *1.1/4
    h2_12h = pivot_12h + range_12h * multiplier * 2  # *1.1/6
    l2_12h = pivot_12h - range_12h * multiplier * 2  # *1.1/6
    h1_12h = pivot_12h + range_12h * multiplier * 1  # *1.1/12
    l1_12h = pivot_12h - range_12h * multiplier * 1  # *1.1/12
    
    # Calculate 12h ADX for regime detection (trending vs ranging)
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Align Camarilla levels to 4h timeframe
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 4h Indicators: Choppiness Index (CHOP) for regime confirmation ===
    def calculate_chop(high, low, close, period=14):
        """Calculate Choppiness Index"""
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        chop = np.zeros(len(high))
        for i in range(len(high)):
            if atr_sum[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50.0  # Neutral
        return chop
    
    chop_4h = calculate_chop(high, low, close)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for 12h indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h4_12h_aligned[i]) or np.isnan(l4_12h_aligned[i]) or
            np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop_4h[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h Regime Filter: ADX > 25 = trending, ADX < 25 = ranging ---
        is_trending = adx_12h_aligned[i] > 25
        is_ranging = adx_12h_aligned[i] < 25
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Price ---
        price = close[i]
        
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
                # Exit on opposite Camarilla touch in ranging markets
                if is_ranging and price <= l3_12h_aligned[i] and volume_spike:
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
                # Exit on opposite Camarilla touch in ranging markets
                if is_ranging and price >= h3_12h_aligned[i] and volume_spike:
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
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trending market logic: Breakout of H3/L3 with volume
        if is_trending:
            # Long: Break above H3 with volume spike
            long_breakout = (price > h3_12h_aligned[i]) and volume_spike
            
            # Short: Break below L3 with volume spike
            short_breakout = (price < l3_12h_aligned[i]) and volume_spike
            
            if long_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_breakout:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        
        # Ranging market logic: Mean revert from H4/L4 with volume
        else:  # is_ranging
            # Long: Pullback to L4 with volume spike (mean reversion long)
            long_reversion = (price <= l4_12h_aligned[i]) and volume_spike
            
            # Short: Pullback to H4 with volume spike (mean reversion short)
            short_reversion = (price >= h4_12h_aligned[i]) and volume_spike
            
            if long_reversion:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_reversion:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals