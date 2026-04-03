#!/usr/bin/env python3
"""
Experiment #1938: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation
HYPOTHESIS: Daily Donchian channel breakouts capture significant momentum moves. 
Weekly HMA(21) filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
Volume confirmation (>1.5x 20-day average) adds conviction to breakouts.
ATR-based stoploss (2.5x ATR) limits downside. Designed for low trade frequency 
(15-25/year) to minimize fee drag while maintaining edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1938_1d_donchian20_1w_hma_vol_v1"
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
    
    # Calculate weekly HMA(21)
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, length):
        if length <= 0:
            return np.full_like(values, np.nan)
        weights = np.arange(1, length + 1, dtype=np.float64)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate WMA for weekly close
    wma_vals = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i >= 20:  # Need 21 for WMA(21)
            wma_vals[i] = wma(close_1w[max(0, i-20):i+1], 21)
    
    # WMA of half length
    wma_half = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i >= half_len - 1:
            wma_half[i] = wma(close_1w[max(0, i-half_len+1):i+1], half_len)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n))
    hma_1w = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if not np.isnan(wma_vals[i]) and not np.isnan(wma_half[i]):
            diff = 2 * wma_half[i] - wma_vals[i]
            # Need sqrt_len values for final WMA
            start_idx = max(0, i - sqrt_len + 1)
            if i >= sqrt_len - 1 and start_idx >= 0:
                # Extract the diff values for WMA calculation
                diff_vals = []
                for j in range(start_idx, i + 1):
                    if j >= 20:  # Ensure we have WMA values
                        diff_2n2 = 2 * wma_half[j] - wma_vals[j]
                        if not np.isnan(diff_2n2):
                            diff_vals.append(diff_2n2)
                if len(diff_vals) >= sqrt_len:
                    weights = np.arange(1, sqrt_len + 1, dtype=np.float64)
                    hma_1w[i] = np.convolve(diff_vals[-sqrt_len:], weights, mode='valid') / weights.sum()
    
    # 1 = bullish trend (price > HMA), -1 = bearish trend (price < HMA)
    trend_1w = np.where(close_1w > hma_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            donchian_high[i] = np.max(high[i-lookback+1:i+1])
            donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:  # Need 20 periods
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_ratio = np.ones(n)
    vol_ratio[19:] = volume[19:] / vol_ma[19:]
    
    # ATR(14) for stoploss
    def calculate_atr(high, low, close, length):
        tr = np.full_like(high, np.nan)
        for i in range(len(high)):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.full_like(high, np.nan)
        for i in range(length-1, len(tr)):
            if not np.isnan(tr[i-length+1:i+1]).any():
                atr[i] = np.mean(tr[i-length+1:i+1])
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    warmup = max(lookback, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss: exit if price moves 2.5*ATR against position
            exit_signal = False
            if position_side > 0:  # Long position
                if price <= entry_price - 2.5 * entry_atr:
                    exit_signal = True
            else:  # Short position
                if price >= entry_price + 2.5 * entry_atr:
                    exit_signal = True
            
            # Additional exit: Donchian opposite touch (mean reversion)
            if not exit_signal:
                if position_side > 0 and price <= donchian_low[i]:
                    exit_signal = True
                elif position_side < 0 and price >= donchian_high[i]:
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
        # Require weekly trend alignment for bias filter
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian Upper AND weekly trend up
            if trend_bias > 0 and price > donchian_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian Lower AND weekly trend down
            elif trend_bias < 0 and price < donchian_low[i]:
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