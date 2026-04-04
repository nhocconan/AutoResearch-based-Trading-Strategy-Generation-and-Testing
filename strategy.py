#!/usr/bin/env python3
"""
Experiment #5030: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: On daily timeframe, Donchian(20) breakouts aligned with weekly HMA trend capture strong momentum with low frequency. Weekly HMA acts as trend filter: only take breakouts in trend direction. Volume > 1.5x average confirms participation. ATR(14) trailing stop (2.0x) manages risk. Designed for 7-25 trades/year on 1d timeframe to minimize fee drag while maintaining statistical significance. Weekly HMA adapts to both bull (rising) and bear (falling) markets, allowing breakouts in prevailing trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5030_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for HMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: HMA(21) for trend ===
    if len(df_1w) >= 21:
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        n = 21
        half_n = n // 2
        sqrt_n = int(np.sqrt(n))
        
        # WMA helper
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / (window * (window + 1) / 2)
        
        # Calculate WMA for full array
        wma_full = np.full(n, np.nan)
        for i in range(n-1, -1, -1):
            if i < n-1:
                wma_full[i] = wma_full[i+1]
            if i >= n-1:
                pass  # Will calculate properly below
        
        # Proper WMA calculation
        wma_vals = np.full(n, np.nan)
        for i in range(n):
            if i >= half_n - 1:
                start = i - half_n + 1
                end = i + 1
                if start >= 0:
                    subset = close[start:end]
                    weights = np.arange(1, len(subset) + 1)
                    wma_vals[i] = np.dot(subset, weights) / (len(subset) * (len(subset) + 1) / 2)
        
        wma_half = np.full(n, np.nan)
        for i in range(n):
            if i >= half_n - 1:
                start = i - half_n + 1
                end = i + 1
                if start >= 0:
                    subset = close[start:end]
                    weights = np.arange(1, len(subset) + 1)
                    wma_half[i] = np.dot(subset, weights) / (len(subset) * (len(subset) + 1) / 2)
        
        wma_full = np.full(n, np.nan)
        for i in range(n):
            if i >= n - 1:
                start = i - n + 1
                end = i + 1
                if start >= 0:
                    subset = close[start:end]
                    weights = np.arange(1, len(subset) + 1)
                    wma_full[i] = np.dot(subset, weights) / (len(subset) * (len(subset) + 1) / 2)
        
        # HMA = WMA(2*WMA_half - WMA_full, sqrt_n)
        hma_raw = 2 * wma_half - wma_full
        hma_vals = np.full(n, np.nan)
        for i in range(n):
            if i >= sqrt_n - 1:
                start = i - sqrt_n + 1
                end = i + 1
                if start >= 0:
                    subset = hma_raw[start:end]
                    weights = np.arange(1, len(subset) + 1)
                    hma_vals[i] = np.dot(subset, weights) / (len(subset) * (len(subset) + 1) / 2)
        
        # Align to 1d timeframe
        hma_aligned = align_htf_to_ltf(prices, df_1w, hma_vals)
        hma_slope = np.diff(hma_aligned, prepend=hma_aligned[0])
    else:
        hma_aligned = np.full(n, np.nan)
        hma_slope = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 21)  # Donchian, Volume MA, ATR, HMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(hma_aligned[i]) or np.isnan(hma_slope[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Trend filter: HMA slope positive for long, negative for short
        trend_long = hma_slope[i] > 0
        trend_short = hma_slope[i] < 0
        
        # Donchian breakout conditions with HMA trend alignment
        # Long: Donchian breakout above high AND uptrend AND volume
        # Short: Donchian breakdown below low AND downtrend AND volume
        breakout_long = (price >= high_roll[i]) and trend_long and vol_confirm
        breakout_short = (price <= low_roll[i]) and trend_short and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals