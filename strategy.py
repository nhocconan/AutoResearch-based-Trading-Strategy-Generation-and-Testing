#!/usr/bin/env python3
"""
Experiment #943: 4h Donchian Breakout + HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture significant momentum moves. 
Trading in direction of 12h HMA(21) trend filters out counter-trend noise. Volume confirmation 
(>1.8x average) ensures institutional participation. ATR-based stoploss (2.5x) manages risk. 
Discrete position sizing (0.30) limits drawdown. Target: 75-200 total trades over 4 years 
(19-50/year) on 4h timeframe. Works in both bull (breakouts) and bear (trend filter avoids 
false breakouts in ranging markets).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_943_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    def hma(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if half_period == 0:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        # Need to align arrays: wma_half starts at index half_period-1, wma_full at period-1
        # We'll compute HMA using pandas for simplicity and proper alignment
        close_s = pd.Series(arr)
        wma_half_s = close_s.ewm(alpha=2/(half_period+1), adjust=False).mean()
        wma_full_s = close_s.ewm(alpha=2/(period+1), adjust=False).mean()
        raw_hma = 2 * wma_half_s - wma_full_s
        hma_values = raw_hma.ewm(alpha=2/(sqrt_period+1), adjust=False).mean()
        return hma_values.values
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(donchian_period, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 20 bars (~3.3d on 4h) to avoid overtrading
            if bars_since_entry > 20:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Determine trend direction from 12h HMA
            # Use previous bar's HMA to avoid look-ahead (already shifted by align_htf_to_ltf)
            hma_trend = hma_12h_aligned[i]
            price_vs_hma = price > hma_trend
            
            # Breakout logic: only trade in direction of trend
            if price > highest_high[i] and price_vs_hma:  # Bullish breakout in uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < lowest_low[i] and not price_vs_hma:  # Bearish breakout in downtrend
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals