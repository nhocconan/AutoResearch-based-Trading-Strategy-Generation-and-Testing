#!/usr/bin/env python3
"""
Experiment #1918: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian channel breakouts capture institutional momentum. 
Weekly HMA(21) filter ensures trades align with major trend, reducing whipsaws in ranging markets.
Volume confirmation (>1.5x 20-day average) adds conviction to breakouts.
ATR-based stoploss (2.5x ATR(14)) manages risk. Designed for low trade frequency (~50-80/4 years) 
to minimize fee drag while maintaining profitability in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1918_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Weekly HMA(21): HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    def hma(arr, period):
        half = period // 2
        sqrt_n = int(np.sqrt(period))
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        # Pad to original length
        wma_half_padded = np.concatenate([np.full(len(arr) - len(wma_half), np.nan), wma_half])
        wma_full_padded = np.concatenate([np.full(len(arr) - len(wma_full), np.nan), wma_full])
        raw_hma = 2 * wma_half_padded - wma_full_padded
        hma_values = wma(raw_hma, sqrt_n)
        # Pad final result
        return np.concatenate([np.full(len(arr) - len(hma_values), np.nan), hma_values])
    
    hma_21_1w = hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === 1d Indicators: Donchian(20), ATR(14), Volume MA(20) ===
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), ATR(14), volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(hma_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss hit
            if position_side > 0:  # Long
                if price <= stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price >= stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Time-based exit: max 20 days holding period
            if bars_since_entry >= 20:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Trend filter: price above/below weekly HMA
        trend_up = price > hma_21_1w_aligned[i]
        trend_down = price < hma_21_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND weekly trend up
            if trend_up and price > donchian_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_price = entry_price - 2.5 * atr[i]  # 2.5x ATR stoploss
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND weekly trend down
            elif trend_down and price < donchian_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_price = entry_price + 2.5 * atr[i]  # 2.5x ATR stoploss
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals