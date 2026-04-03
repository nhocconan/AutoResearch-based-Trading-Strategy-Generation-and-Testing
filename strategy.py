#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Price breaking 12h Donchian(20) channels with alignment to weekly HMA trend captures momentum with institutional bias. Volume confirmation (>1.8x) filters false breakouts. Weekly HMA provides multi-timeframe trend filter that works in bull (continuation) and bear (mean reversion at extremes). Discrete sizing (0.25) controls fee drag. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_008_12h_donchian20_1w_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on weekly close
    if len(df_1w) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        n_hma = 21
        half_n = n_hma // 2
        sqrt_n = int(np.sqrt(n_hma))
        
        # WMA helper
        def wma(arr, window):
            if len(arr) < window:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_1w = df_1w['close'].values
        wma_full = wma(close_1w, n_hma)
        wma_half = wma(close_1w, half_n)
        
        # Handle array lengths
        wma_2x_half = np.full_like(close_1w, np.nan)
        wma_2x_half[half_n-1:len(wma_half)+half_n-1] = 2 * wma_half
        
        # Align lengths for subtraction
        min_len = min(len(wma_full), len(wma_2x_half))
        diff = np.full_like(close_1w, np.nan)
        diff[:min_len] = wma_2x_half[:min_len] - wma_full[:min_len]
        
        hma_vals = wma(diff, sqrt_n)
        hma_1w = np.full_like(close_1w, np.nan)
        start_idx = len(diff) - len(hma_vals)
        if start_idx >= 0 and start_idx < len(hma_1w):
            hma_1w[start_idx:start_idx+len(hma_vals)] = hma_vals
        
        # Trend: price above HMA = bullish, below = bearish
        hma_trend = (close_1w > hma_1w).astype(float)
        
        # Align to 12h timeframe
        hma_trend_aligned = align_htf_to_ltf(prices, df_1w, hma_trend)
    else:
        # Not enough data - neutral trend
        hma_trend_aligned = np.zeros(n)
    
    # === 12h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Exit Logic: ATR-based stoploss (using 2.5*ATR for wider stops on 12h) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr = np.zeros(i+1)
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr[0] = high[0] - low[0]
                atr_val = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            else:
                atr_val = 0.0
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr_val
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~3 days on 12h)
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND bullish HMA trend
            if breakout_up and hma_trend_aligned[i] > 0.5:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish HMA trend
            elif breakout_down and hma_trend_aligned[i] < 0.5:
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