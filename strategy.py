#!/usr/bin/env python3
"""
Experiment #2074: 1h Donchian(20) breakout + 4h/1d trend filter + volume confirmation + ATR stoploss
HYPOTHESIS: 1h Donchian breakouts with HTF (4h/1d) alignment capture swing moves while avoiding counter-trend noise.
- Primary: 1h Donchian(20) breakout with volume > 1.5x 20-bar average
- HTF: 4h HMA(21) and 1d HMA(21) trend filter (only trade when both agree)
- Exit: ATR(14) trailing stop (2*ATR) or opposite Donchian channel touch
- Session filter: 08-20 UTC to avoid low-liquidity hours
Target: 60-150 total trades over 4 years (15-37/year) with strict entry filters to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2074_1h_donchian20_4h_1d_hma_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for HMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h HMA(21)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
    
    # Calculate WMA for close_4h
    wma_full_4h = np.array([np.nan] * len(close_4h))
    wma_half_4h = np.array([np.nan] * len(close_4h))
    
    for i in range(20, len(close_4h)):  # WMA(21)
        wma_full_4h[i] = np.mean(close_4h[i-20:i+1] * np.arange(1, 22))
    for i in range(half_len-1, len(close_4h)):
        wma_half_4h[i] = np.mean(close_4h[i-half_len+1:i+1] * np.arange(1, half_len+1))
    
    # HMA = WMA(2*WMA_half - WMA_full, sqrt_len)
    wma_diff_4h = 2 * wma_half_4h - wma_full_4h
    hma_4h = np.array([np.nan] * len(close_4h))
    for i in range(sqrt_len-1, len(close_4h)):
        if i >= half_len-1 and not np.isnan(wma_diff_4h[i]):
            hma_4h[i] = np.mean(wma_diff_4h[i-sqrt_len+1:i+1] * np.arange(1, sqrt_len+1))
    
    # Trend: 1 if close > HMA, -1 otherwise
    trend_4h = np.where(close_4h > hma_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA(21)
    wma_full_1d = np.array([np.nan] * len(close_1d))
    wma_half_1d = np.array([np.nan] * len(close_1d))
    
    for i in range(20, len(close_1d)):  # WMA(21)
        wma_full_1d[i] = np.mean(close_1d[i-20:i+1] * np.arange(1, 22))
    for i in range(half_len-1, len(close_1d)):
        wma_half_1d[i] = np.mean(close_1d[i-half_len+1:i+1] * np.arange(1, half_len+1))
    
    # HMA = WMA(2*WMA_half - WMA_full, sqrt_len)
    wma_diff_1d = 2 * wma_half_1d - wma_full_1d
    hma_1d = np.array([np.nan] * len(close_1d))
    for i in range(sqrt_len-1, len(close_1d)):
        if i >= half_len-1 and not np.isnan(wma_diff_1d[i]):
            hma_1d[i] = np.mean(wma_diff_1d[i-sqrt_len+1:i+1] * np.arange(1, sqrt_len+1))
    
    # Trend: 1 if close > HMA, -1 otherwise
    trend_1d = np.where(close_1d > hma_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size - conservative to manage drawdown
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC only ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower Donchian (mean reversion)
                elif price <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper Donchian (mean reversion)
                elif price >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require BOTH 4h and 1d trend alignment for stronger bias filter
        trend_bias_4h = trend_4h_aligned[i]
        trend_bias_1d = trend_1d_aligned[i]
        
        # Only trade when both timeframes agree
        if trend_bias_4h == trend_bias_1d:
            trend_bias = trend_bias_4h  # Either 1 or -1
        else:
            trend_bias = 0  # No clear trend - stay out
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike and trend_bias != 0:
            # Long entry: price breaks above upper Donchian AND HTF trend up
            if trend_bias > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND HTF trend down
            elif trend_bias < 0 and price < donchian_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals