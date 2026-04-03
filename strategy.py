#!/usr/bin/env python3
"""
Experiment #1866: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture strong momentum moves. Combined with 1d HMA trend filter and volume confirmation (>1.5x average), this strategy enters only during strong trending conditions. ATR-based stoploss (2.5x ATR) manages risk. Works in both bull and bear markets by following the 1d trend direction. Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing of 0.30.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1866_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d HMA(21) for trend direction
    def calculate_hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean().values
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw_hma = 2 * wma2 - wma1
        hma = pd.Series(raw_hma).ewm(span=sqrt, adjust=False).mean().values
        return hma
    
    hma_21_1d = calculate_hma(close_1d, 21)
    trend_1d = np.where(close_1d > hma_21_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian Channel(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # === 4h Indicators: HMA(21) for entry confirmation ===
    hma_21 = calculate_hma(close, 21)
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20) and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(hma_21[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or reverse signal ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss conditions
            stoploss_hit = False
            if position_side > 0:  # Long position
                if price <= entry_price - 2.5 * entry_atr:
                    stoploss_hit = True
            else:  # Short position
                if price >= entry_price + 2.5 * entry_atr:
                    stoploss_hit = True
            
            # Exit conditions
            exit_signal = False
            if stoploss_hit:
                exit_signal = True
            elif position_side > 0 and price < donchian_lower[i]:  # Long exit on lower band break
                exit_signal = True
            elif position_side < 0 and price > donchian_upper[i]:  # Short exit on upper band break
                exit_signal = True
            elif trend_1d_aligned[i] != position_side:  # Exit if 1d trend flips
                exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Require volume confirmation: volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        breakout_up = price > donchian_upper[i-1]  # Break above upper band
        breakout_down = price < donchian_lower[i-1]  # Break below lower band
        
        if volume_spike:
            if trend_bias > 0 and breakout_up and price > hma_21[i]:  # Long: uptrend + upper breakout + above HMA
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif trend_bias < 0 and breakout_down and price < hma_21[i]:  # Short: downtrend + lower breakout + below HMA
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals