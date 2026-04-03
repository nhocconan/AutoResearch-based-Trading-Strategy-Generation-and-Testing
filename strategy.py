#!/usr/bin/env python3
"""
Experiment #293: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Price breaking 4h Donchian(20) channels with 12h HMA(21) trend confirmation and volume spike captures strong momentum moves in both bull and bear markets. The 12h HMA filters for higher-timeframe trend alignment, reducing false breakouts. Volume confirmation ensures institutional participation. ATR-based stoploss manages risk. Discrete sizing (0.25) minimizes fee drag. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_293_4h_donchian20_12h_hma21_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA(21) trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    hma_period = 21
    
    # Calculate HMA(21) on 12h close
    half_period = hma_period // 2
    sqrt_period = int(np.sqrt(hma_period))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1, dtype=np.float64)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate HMA: WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    wma_half = np.array([np.nan] * len(df_12h))
    wma_full = np.array([np.nan] * len(df_12h))
    
    for i in range(half_period, len(df_12h)):
        wma_half[i] = wma(df_12h['close'].values[i - half_period + 1:i + 1], half_period)
    
    for i in range(hma_period, len(df_12h)):
        wma_full[i] = wma(df_12h['close'].values[i - hma_period + 1:i + 1], hma_period)
    
    # HMA = WMA(2*WMA_half - WMA_full, sqrt_period)
    hma_12h = np.array([np.nan] * len(df_12h))
    for i in range(hma_period, len(df_12h)):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2 * wma_half[i] - wma_full[i]
            start_idx = i - hma_period + 1
            end_idx = i + 1
            if start_idx >= 0 and end_idx <= len(df_12h):
                wma_diff = wma(df_12h['close'].values[start_idx:end_idx], sqrt_period)
                if not np.isnan(wma_diff):
                    hma_12h[i] = wma_diff
    
    # Align HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian(20) channels ===
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # Enough for Donchian(20), ATR(14), Vol MA(20), HMA calc
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- 12h HMA Trend Filter ---
        # Long only when price above 12h HMA (bullish trend)
        # Short only when price below 12h HMA (bearish trend)
        long_trend = price > hma_12h_aligned[i]
        short_trend = price < hma_12h_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
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
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up AND bullish 12h HMA trend
            if breakout_up and long_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down AND bearish 12h HMA trend
            elif breakout_down and short_trend:
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