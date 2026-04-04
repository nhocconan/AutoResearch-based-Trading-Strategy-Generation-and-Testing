#!/usr/bin/env python3
"""
Experiment #4173: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: 4h Donchian breakouts aligned with 12h Hull Moving Average trend capture medium-term momentum while reducing whipsaw. The 12h HMA provides smoother trend direction than EMA/SMA, and volume confirmation (>1.4x) ensures breakout validity. ATR-based trailing stop (2.5x) manages risk. Designed to work in both bull and bear markets by following the 12h trend. Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4173_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        # Calculate Hull Moving Average (HMA) on 12h close
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        close_12h = df_12h['close'].values
        half_len = len(close_12h) // 2
        sqrt_len = int(np.sqrt(len(close_12h)))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        # Handle edge cases for WMA calculation
        wma_half = np.full_like(close_12h, np.nan)
        wma_full = np.full_like(close_12h, np.nan)
        
        if half_len > 0 and len(close_12h) >= half_len:
            wma_half_vals = wma(close_12h, half_len)
            start_idx = half_len - 1
            wma_half[start_idx:start_idx + len(wma_half_vals)] = wma_half_vals
        
        if len(close_12h) >= 20:  # Using 20 as period for WMA
            wma_full_vals = wma(close_12h, 20)
            start_idx = 19
            wma_full[start_idx:start_idx + len(wma_full_vals)] = wma_full_vals
        
        raw_hma = 2 * wma_half - wma_full
        hma_12h = np.full_like(close_12h, np.nan)
        if sqrt_len > 0 and len(raw_hma) >= sqrt_len:
            wma_hma_vals = wma(raw_hma[~np.isnan(raw_hma)], sqrt_len) if np.sum(~np.isnan(raw_hma)) >= sqrt_len else np.array([])
            if len(wma_hma_vals) > 0:
                start_idx = np.where(~np.isnan(raw_hma))[0][sqrt_len - 1] if np.sum(~np.isnan(raw_hma)) >= sqrt_len else 0
                end_idx = start_idx + len(wma_hma_vals)
                hma_12h[start_idx:end_idx] = wma_hma_vals
        
        hma_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(lookback_dc + 1, 20 + 5, 20 + 5, 14 + 5)  # DC lookback, vol MA buffer, HMA buffer, ATR buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.4x average) to filter noise
        volume_spike = vol_ratio[i] > 1.4
        
        if volume_spike:
            # Donchian breakout logic
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # 12h HMA trend filter: price above HMA = bullish bias, below HMA = bearish bias
            above_hma = price > hma_aligned[i]
            below_hma = price < hma_aligned[i]
            
            # Long conditions: Donchian breakout up + price above 12h HMA
            long_entry = breakout_up and above_hma
            
            # Short conditions: Donchian breakout down + price below 12h HMA
            short_entry = breakout_down and below_hma
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
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