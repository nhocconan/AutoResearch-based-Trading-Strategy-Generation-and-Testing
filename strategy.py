#!/usr/bin/env python3
"""
Experiment #357: 4h Donchian Breakout + 1d Volume Spike + Chop Regime Filter

HYPOTHESIS: Donchian(20) breakouts on 4h confirmed by 1d volume spike (>2x 20-period average)
and only traded when market is trending (Choppiness Index < 38.2). Uses 1d HTF for volume
and chop regime to avoid false breakouts in sideways markets. ATR-based stoploss manages risk.
Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag. Works in both bull
(breakouts) and bear (breakdowns) markets by capturing strong momentum moves after
consolidation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_1d_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) - helps reduce noise
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 1d data for volume confirmation and chop regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # 1d volume ratio (current vs 20-period MA)
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
        
        # 1d Choppiness Index (14-period)
        def calculate_chop(high_arr, low_arr, close_arr, period=14):
            n_local = len(close_arr)
            chop = np.full(n_local, np.nan)
            if n_local < period * 2:
                return chop
            tr = np.zeros(n_local)
            tr[0] = high_arr[0] - low_arr[0]
            for i in range(1, n_local):
                tr[i] = max(high_arr[i] - low_arr[i], 
                           abs(high_arr[i] - close_arr[i-1]), 
                           abs(low_arr[i] - close_arr[i-1]))
            atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
            max_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
            min_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
            chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(period)
            return chop
        
        chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
        chop_1d_aligned = np.full(n, 50.0)  # neutral chop
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
        # Warmup period: use expanding window
        for i in range(20):
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Chop Regime Filter: Only trade when trending (CHOP < 38.2) ---
        if chop_1d_aligned[i] >= 38.2:
            signals[i] = 0.0
            continue
        
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
                # Take profit at Donchian Low (trailing stop)
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
                # Take profit at Donchian High (trailing stop)
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian High with volume confirmation
        long_condition = (
            close[i] > donchian_high[i] and 
            vol_ratio_1d_aligned[i] > 2.0
        )
        
        # Short: Break below Donchian Low with volume confirmation
        short_condition = (
            close[i] < donchian_low[i] and 
            vol_ratio_1d_aligned[i] > 2.0
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