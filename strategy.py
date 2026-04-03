#!/usr/bin/env python3
"""
Experiment #061: 4h Donchian(20) Breakout + HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: Donchian(20) breakouts from 4h timeframe with volume confirmation (>1.5x average)
and HMA(21) trend filter capture strong momentum moves. ATR-based stoploss (2.5x) limits drawdown.
Works in both bull/bear regimes by only taking breakouts in direction of HMA(21) trend.
Target: 75-200 trades over 4 years on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_061_4h_donchian_hma_vol_atr_v1"
timeframe = "4h"
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
        """Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA calculation
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        
        wma_half = np.array([wma(arr[i:i+half_period], half_period) 
                            if i+half_period <= len(arr) else np.nan 
                            for i in range(len(arr))])
        wma_full = np.array([wma(arr[i:i+period], period) 
                            if i+period <= len(arr) else np.nan 
                            for i in range(len(arr))])
        
        # Handle edge cases with proper alignment
        wma_half = np.concatenate([np.full(half_period-1, np.nan), wma_half])[:len(arr)]
        wma_full = np.concatenate([np.full(period-1, np.nan), wma_full])[:len(arr)]
        
        raw_hma = 2 * wma_half - wma_full
        hma = np.array([wma(raw_hma[i:i+sqrt_period], sqrt_period) 
                       if i+sqrt_period <= len(raw_hma) else np.nan 
                       for i in range(len(raw_hma))])
        hma = np.concatenate([np.full(sqrt_period-1, np.nan), hma])[:len(arr)]
        return hma
    
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
    entry_atr = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume spike threshold
        
        # --- Trend Filter ---
        is_uptrend = hma_1d_aligned[i] > hma_1d_aligned[i-1] if i > 0 else False
        is_downtrend = hma_1d_aligned[i] < hma_1d_aligned[i-1] if i > 0 else False
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss: 2.5 * ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.5 * entry_atr:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price > entry_price + 2.5 * entry_atr:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Exit on Donchian break in opposite direction with volume
            if position_side > 0:  # Long
                if low[i] < donchian_lower[i-1] and vol_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if high[i] > donchian_upper[i-1] and vol_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 1 bar
            if bars_since_entry < 1:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: Donchian upper break + volume + uptrend
        if high[i] > donchian_upper[i-1] and vol_spike and is_uptrend:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = atr[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Donchian lower break + volume + downtrend
        elif low[i] < donchian_lower[i-1] and vol_spike and is_downtrend:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = atr[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals