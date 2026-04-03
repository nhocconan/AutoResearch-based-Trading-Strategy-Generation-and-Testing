#!/usr/bin/env python3
"""
Experiment #005: 12h Camarilla Pivot + 1d Volume Spike + Chop Regime Filter

HYPOTHESIS: Camarilla pivot levels on 12h derived from 1d OHLC provide institutional support/resistance. 
Trading breakouts of these levels with 1d volume confirmation (>2.0x average) and filtering by 
choppiness regime (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend) avoids false breakouts 
in ranging markets. Uses ATR-based stoploss (2.5x) and minimum 4-bar holding period. 
Target: 75-150 trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_005_12h_camarilla_pivot_1d_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots, volume MA, chop regime ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels for 12h timeframe
    # Camarilla: based on previous day's OHLC
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    pivot = (h_1d + l_1d + c_1d) / 3.0
    range_1d = h_1d - l_1d
    
    # Camarilla levels
    camarilla_h4 = pivot + (range_1d * 1.1 / 2.0)  # Resistance 4
    camarilla_l4 = pivot - (range_1d * 1.1 / 2.0)  # Support 4
    camarilla_h3 = pivot + (range_1d * 1.1 / 4.0)  # Resistance 3
    camarilla_l3 = pivot - (range_1d * 1.1 / 4.0)  # Support 3
    camarilla_h2 = pivot + (range_1d * 1.1 / 6.0)  # Resistance 2
    camarilla_l2 = pivot - (range_1d * 1.1 / 6.0)  # Support 2
    camarilla_h1 = pivot + (range_1d * 1.1 / 12.0) # Resistance 1
    camarilla_l1 = pivot - (range_1d * 1.1 / 12.0) # Support 1
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 for completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.full_like(vol_1d, 1.0)
    vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1d Indicators: Choppiness Index (CHOP) for regime filter ===
    def calculate_chop(high, low, close, period=14):
        """Choppiness Index: higher = ranging, lower = trending"""
        atr_sum = np.zeros_like(high)
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Wilder's smoothing (equivalent to RMA)
        atr_sum[period-1] = np.sum(tr[1:period])  # Seed
        for i in range(period, len(high)):
            atr_sum[i] = (atr_sum[i-1] * (period-1) + tr[i]) / period
        # Avoid division by zero
        max_min = np.zeros_like(high)
        for i in range(len(high)):
            if i >= period:
                max_high = np.max(high[i-period+1:i+1])
                min_low = np.min(low[i-period+1:i+1])
                max_min[i] = max_high - min_low
        chop = np.full_like(high, 50.0)  # Default neutral
        valid = (max_min > 0) & (i >= period)
        chop[valid] = 100 * np.log10(atr_sum[valid] / max_min[valid] / np.sqrt(period)) / np.log10(100)
        return chop
    
    chop_1d = calculate_chop(h_1d, l_1d, c_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr_12h = np.zeros(n)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Warmup for 1d indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filter: Choppiness Index ---
        chop = chop_1d_aligned[i]
        is_ranging = chop > 61.8   # High chop = ranging (mean revert)
        is_trending = chop < 38.2  # Low chop = trending (trend follow)
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Camarilla Breakout Conditions ---
        breakout_h4 = high[i] > camarilla_h4_aligned[i-1]  # Break above H4
        breakout_l4 = low[i] < camarilla_l4_aligned[i-1]   # Break below L4
        breakout_h3 = high[i] > camarilla_h3_aligned[i-1]  # Break above H3
        breakout_l3 = low[i] < camarilla_l3_aligned[i-1]   # Break below L3
        
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
                # Exit on opposite Camarilla breakout (contrarian exit)
                if breakout_l4 and volume_spike:
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
                # Exit on opposite Camarilla breakout (contrarian exit)
                if breakout_h4 and volume_spike:
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
        # In trending regime: breakout of H3/L3 with volume
        if is_trending:
            # Long: Break above H3 AND volume spike
            if breakout_h3 and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Break below L3 AND volume spike
            elif breakout_l3 and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        # In ranging regime: mean reversion at H4/L4 levels
        elif is_ranging:
            # Long: Price rejects L4 support (breaks below then closes above)
            # Short: Price rejects H4 resistance (breaks above then closes below)
            if low[i] < camarilla_l4_aligned[i] and close[i] > camarilla_l4_aligned[i] and volume_spike:
                # Bullish rejection at L4
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif high[i] > camarilla_h4_aligned[i] and close[i] < camarilla_h4_aligned[i] and volume_spike:
                # Bearish rejection at H4
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # Neutral chop zone, no trade
            signals[i] = 0.0
    
    return signals