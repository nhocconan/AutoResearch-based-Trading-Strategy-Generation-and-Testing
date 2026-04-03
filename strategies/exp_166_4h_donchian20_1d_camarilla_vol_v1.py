#!/usr/bin/env python3
"""
Experiment #157: 4h Donchian Breakout + Volume Spike + Chop Regime Filter
HYPOTHESIS: 4h Donchian channel breakouts combined with volume confirmation and choppiness regime filter capture strong trending moves while avoiding whipsaws in sideways markets. Works in bull/bear regimes by requiring volume confirmation for breakout validity. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_157_4h_donchian_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for choppiness regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_period = 14
    
    # True Range for 1d
    tr_1d = np.zeros(len(close_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Sum of TR over atr_period
    tr_sum = pd.Series(tr_1d).rolling(window=atr_period, min_periods=atr_period).sum().values
    
    # Highest high and lowest low over atr_period
    hh = pd.Series(high_1d).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low_1d).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Choppiness Index
    chop = np.zeros(len(close_1d))
    denominator = hh - ll
    valid = (denominator > 0) & ~np.isnan(denominator)
    chop[valid] = 100 * np.log10(tr_sum[valid] / denominator[valid]) / np.log10(atr_period)
    chop[~valid] = 50.0  # neutral when invalid
    
    # Align to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_up_1w = close_1w > ema50_1w
    trend_down_1w = close_1w < ema50_1w
    
    # Align to 4h timeframe
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(trend_down_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Chop Regime Filter: Only trade when trending (CHOP < 38.2) or ranging (CHOP > 61.8) ---
        chop_value = chop_aligned[i]
        chop_trending = chop_value < 38.2
        chop_ranging = chop_value > 61.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
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
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian HIGH, volume spike, 1w uptrend, trending OR ranging regime
        if (close[i] > donchian_high[i-1] and  # break above previous period's high
            volume_spike and 
            trend_up_1w_aligned[i] and
            (chop_trending or chop_ranging)):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian LOW, volume spike, 1w downtrend, trending OR ranging regime
        elif (close[i] < donchian_low[i-1] and  # break below previous period's low
              volume_spike and 
              trend_down_1w_aligned[i] and
              (chop_trending or chop_ranging)):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals