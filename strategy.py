#!/usr/bin/env python3
"""
Experiment #087: 6h Donchian(20) breakout + 1d/1w pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with daily/weekly pivot structure capture institutional order flow. 
Daily pivot provides intraday trend bias, weekly pivot captures longer-term structure. Volume confirmation (>1.8x average) ensures breakout legitimacy. 
ATR stoploss (2.0x) and minimum holding period (4 bars) reduce churn. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_087_6h_donchian20_1d_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for daily pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard floor trader pivots)
    def calculate_pivot(high, low, close):
        pivot = (high + low + close) / 3.0
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        return pivot, r1, r2, r3, s1, s2, s3
    
    # Calculate pivots for each 1d bar
    pivots = np.zeros((len(df_1d), 7))  # [pivot, r1, r2, r3, s1, s2, s3]
    for i in range(len(df_1d)):
        pivots[i] = calculate_pivot(df_1d['high'].iloc[i], df_1d['low'].iloc[i], df_1d['close'].iloc[i])
    
    # Align pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivots[:, 0])
    r1_6h = align_htf_to_ltf(prices, df_1d, pivots[:, 1])
    r2_6h = align_htf_to_ltf(prices, df_1d, pivots[:, 2])
    r3_6h = align_htf_to_ltf(prices, df_1d, pivots[:, 3])
    s1_6h = align_htf_to_ltf(prices, df_1d, pivots[:, 4])
    s2_6h = align_htf_to_ltf(prices, df_1d, pivots[:, 5])
    s3_6h = align_htf_to_ltf(prices, df_1d, pivots[:, 6])
    
    # === HTF: 1w data for weekly pivot bias ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    weekly_pivots = np.zeros((len(df_1w), 7))
    for i in range(len(df_1w)):
        weekly_pivots[i] = calculate_pivot(df_1w['high'].iloc[i], df_1w['low'].iloc[i], df_1w['close'].iloc[i])
    
    # Align weekly pivot to 6h timeframe (use weekly pivot as bias filter)
    wp_6h = align_htf_to_ltf(prices, df_1w, weekly_pivots[:, 0])  # weekly pivot
    wr1_6h = align_htf_to_ltf(prices, df_1w, weekly_pivots[:, 1])  # weekly R1
    ws1_6h = align_htf_to_ltf(prices, df_1w, weekly_pivots[:, 4])  # weekly S1
    
    # === 6h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 60  # Warmup for Donchian and pivot stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(wp_6h[i]) or np.isnan(wr1_6h[i]) or np.isnan(ws1_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Pivot Bias: Determine trend from daily and weekly pivot ---
        # Daily bias: price above/below daily pivot
        daily_bias_up = price > pivot_6h[i]
        daily_bias_down = price < pivot_6h[i]
        
        # Weekly bias: price above/below weekly pivot (stronger filter)
        weekly_bias_up = price > wp_6h[i]
        weekly_bias_down = price < wp_6h[i]
        
        # Combined bias: require both daily and weekly agreement for stronger signal
        bias_up = daily_bias_up and weekly_bias_up
        bias_down = daily_bias_down and weekly_bias_down
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]  # Break above upper channel
        breakout_down = low[i] < donch_lower[i-1]  # Break below lower channel
        
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
                # Exit on opposite Donchian breakout with volume (profit taking)
                if breakout_down and volume_spike:
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
                # Exit on opposite Donchian breakout with volume (profit taking)
                if breakout_up and volume_spike:
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
        # Only trade when breakout aligns with daily AND weekly pivot bias
        if bias_up:
            # Long: Donchian breakout up AND volume spike AND daily/weekly bias up
            if breakout_up and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        elif bias_down:
            # Short: Donchian breakout down AND volume spike AND daily/weekly bias down
            if breakout_down and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # No clear bias, do not trade
            signals[i] = 0.0
    
    return signals