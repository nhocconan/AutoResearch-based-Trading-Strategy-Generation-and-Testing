#!/usr/bin/env python3
"""
Experiment #723: 4h Donchian20 + 12h HMA Trend + Volume Spike + Chop Filter
HYPOTHESIS: 4h Donchian(20) breakouts filtered by 12h HMA(21) trend direction and volume confirmation (>1.5x average) 
captures strong momentum moves while avoiding choppy markets. Uses discrete sizing (0.25) and ATR-based stoploss (2.0).
Designed for both bull and bear markets: trend filter prevents counter-trend entries, volume confirms institutional interest.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_723_4h_donchian20_12h_hma_vol_chop_v1"
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
    
    # Calculate HMA(21) on 12h: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    def hma(arr, period):
        half = period // 2
        sqrt_n = int(np.sqrt(period))
        if half == 0:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        if len(wma_half) == 0 or len(wma_full) == 0:
            return np.full_like(arr, np.nan)
        wma_diff = 2 * wma_half - wma_full
        return wma(wma_diff, sqrt_n)
    
    # Pad HMA to match original length
    hma_raw = hma(close_12h, 21)
    hma_12h = np.full_like(close_12h, np.nan)
    if len(hma_raw) > 0:
        start_idx = len(close_12h) - len(hma_raw)
        hma_12h[start_idx:] = hma_raw
    
    # HMA trend: 1 = rising (bullish), -1 = falling (bearish)
    hma_trend = np.zeros_like(close_12h)
    hma_trend[1:] = np.where(hma_12h[1:] > hma_12h[:-1], 1, -1)
    hma_trend[0] = 0
    
    # Align HMA trend to 4h timeframe
    hma_trend_aligned = align_htf_to_ltf(prices, df_12h, hma_trend)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
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
    
    # === 4h Indicators: Choppiness Index (14) for regime filter ===
    def choppiness_index(high, low, close, period=14):
        atr_sum = np.zeros_like(close)
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr[0] = high[0] - low[0]
        atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        chop = np.ones_like(close) * 50  # default to neutral
        mask = (atr_sum > 0) & (hh > ll)
        chop[mask] = 100 * np.log10(atr_sum[mask] / (hh[mask] - ll[mask])) / np.log10(period)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20, 14)  # sufficient for Donchian, volume MA, chop
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_trend_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        # Chop filter: avoid extreme chop (chop > 61.8) and extreme trending (chop < 38.2) - wait for normalization
        chop_filter = (chop[i] >= 38.2) & (chop[i] <= 61.8)
        
        if volume_spike and chop_filter:
            # Get trend from 12h HMA
            trend = hma_trend_aligned[i]
            
            # Long: price breaks above Donchian upper AND HMA trend bullish
            if high[i] > donchian_high[i] and trend > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND HMA trend bearish
            elif low[i] < donchian_low[i] and trend < 0:
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