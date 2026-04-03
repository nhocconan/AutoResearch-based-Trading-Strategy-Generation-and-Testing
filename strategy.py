#!/usr/bin/env python3
"""
Experiment #347: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts on 6h timeframe, filtered by weekly pivot direction (from 1w HTF) 
and confirmed by 1d volume spike, captures high-probability continuation moves in both bull and bear markets. 
Weekly pivot provides structural bias, Donchian breakout captures momentum, volume confirms institutional 
participation. Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while avoiding 
whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for weekly pivot direction (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot and bias (bullish if close > pivot, bearish if close < pivot)
    weekly_pivot = np.full(n, np.nan)
    weekly_bias = np.zeros(n)  # 1 = bullish, -1 = bearish, 0 = neutral
    
    if len(df_1w) >= 1:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Weekly pivot = (H + L + C) / 3
        pivot_1w = (high_1w + low_1w + close_1w) / 3.0
        
        # Align weekly pivot to 6h bars
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
        
        # Determine weekly bias: 1 if close > pivot, -1 if close < pivot
        for i in range(n):
            wp = weekly_pivot_aligned[i]
            if not np.isnan(wp):
                if close_1w[-1] > wp:  # Use latest weekly close for bias
                    weekly_bias[i] = 1
                elif close_1w[-1] < wp:
                    weekly_bias[i] = -1
                else:
                    weekly_bias[i] = 0
    
    # === 6h Indicators ===
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        for i in range(20, n+1):
            donchian_high[i-1] = np.max(high[i-20:i])
            donchian_low[i-1] = np.min(low[i-20:i])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) if 'weekly_pivot_aligned' in locals() else True or
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Bias --- 
        bias = weekly_bias[i] if 'weekly_bias' in locals() else 0
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian Low (trailing exit)
                if close[i] <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian High (trailing exit)
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above weekly high AND bullish weekly bias AND volume spike
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above 20-period high
            bias > 0 and                     # Bullish weekly bias
            volume_spike                     # Volume confirmation
        )
        
        # Short: Donchian breakdown below weekly low AND bearish weekly bias AND volume spike
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below 20-period low
            bias < 0 and                     # Bearish weekly bias
            volume_spike                     # Volume confirmation
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals