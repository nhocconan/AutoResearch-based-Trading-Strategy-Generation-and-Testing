#!/usr/bin/env python3
"""
Experiment #5668: 12h Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts with volume > 1.8x average and aligned 
with 1w HMA(21) trend (price above HMA = bullish bias, below = bearish bias) capture 
high-probability trend continuation moves. Weekly HMA provides smooth directional filter 
from higher timeframe that works in both bull and bear markets by avoiding entries against 
the major trend. Volume confirms breakout strength. ATR trailing stop (2.5x) manages risk. 
Discrete sizing (0.25) minimizes fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5668_12h_donchian20_1w_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1w data for HMA trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA(21) on weekly close
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # WMA function
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        close_1w = df_1w['close'].values.astype(np.float64)
        wma_half = np.array([wma(close_1w[i:i+half_len], half_len) if i+half_len <= len(close_1w) else np.nan 
                            for i in range(len(close_1w))])
        wma_full = np.array([wma(close_1w[i:i+21], 21) if i+21 <= len(close_1w) else np.nan 
                            for i in range(len(close_1w))])
        wma_sqrt = np.array([wma(close_1w[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(close_1w) else np.nan 
                            for i in range(len(close_1w))])
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        hma_1w = np.array([wma(2*wma_half[i] - wma_full[i], sqrt_len) if not (np.isnan(wma_half[i]) or np.isnan(wma_full[i])) else np.nan 
                          for i in range(len(close_1w))])
        hma_1w = np.concatenate([np.full(half_len-1, np.nan), hma_1w[:len(close_1w)-half_len+1]])
    else:
        hma_1w = np.full(len(df_1w), np.nan)
    
    # Align 1w HMA to 12h timeframe
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 12h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 21)  # Donchian, volume avg, ATR, HMA lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below 1w HMA (trend change)
                if price <= stop_price or price <= hma_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above 1w HMA (trend change)
                if price >= stop_price or price >= hma_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8
        
        # Weekly HMA trend bias: long above HMA, short below HMA
        long_bias = price > hma_1w_aligned[i]
        short_bias = price < hma_1w_aligned[i]
        
        # Entry conditions: breakout in direction of weekly HMA trend with volume
        long_setup = breakout_up and volume_confirmed and long_bias
        short_setup = breakout_down and volume_confirmed and short_bias
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals