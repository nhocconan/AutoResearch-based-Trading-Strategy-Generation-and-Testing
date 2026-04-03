#!/usr/bin/env python3
"""
Experiment #065: 12h Donchian Breakout + Volume + HMA Trend + ATR Stoploss
HYPOTHESIS: 12h Donchian(20) breakouts with volume confirmation (>1.5x average) and 
1d HMA(21) trend filter capture strong momentum moves. ATR-based stoploss (2.5x) 
limits drawdown. Works in both bull/bear markets by using 1d HMA for trend direction.
Target: 50-150 trades over 4 years on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_065_12h_donchian_volume_hma_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: HMA(21) for trend direction ===
    def calculate_hma(arr, period):
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / (window * (window + 1) / 2)
        
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        
        wma_half = np.array([np.nan] * (half_period - 1) + 
                           [wma(arr[i:i+half_period], half_period) 
                            for i in range(len(arr) - half_period + 1)])
        wma_full = np.array([np.nan] * (period - 1) + 
                           [wma(arr[i:i+period], period) 
                            for i in range(len(arr) - period + 1)])
        
        raw_hma = 2 * wma_half - wma_full
        hma = np.array([np.nan] * (sqrt_period - 1) + 
                      [wma(raw_hma[i:i+sqrt_period], sqrt_period) 
                       for i in range(len(raw_hma) - sqrt_period + 1)])
        
        # Pad to original length
        result = np.full_like(arr, np.nan)
        start_idx = period - 1
        end_idx = start_idx + len(hma)
        if end_idx <= len(arr):
            result[start_idx:end_idx] = hma
        return result
    
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 12h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 12h Indicators: ATR(14) for stoploss ===
    def calculate_atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # Warmup for Donchian, volume, ATR, and HMA stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(hma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume spike threshold
        
        # --- Trend Filter: 1d HMA ---
        is_uptrend = price > hma_1d_aligned[i]
        is_downtrend = price < hma_1d_aligned[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Donchian breakout in opposite direction as exit signal
            if position_side > 0:  # Long exit
                if price < donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short exit
                if price > donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: price breaks above Donchian upper with volume and uptrend
        if price > donchian_upper[i-1] and vol_spike and is_uptrend:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        # Short: price breaks below Donchian lower with volume and downtrend
        elif price < donchian_lower[i-1] and vol_spike and is_downtrend:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals