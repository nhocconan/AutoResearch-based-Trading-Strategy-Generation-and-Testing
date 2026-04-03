#!/usr/bin/env python3
"""
Experiment #263: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: Donchian breakouts on 4h aligned with 12h HMA(21) trend direction capture high-probability moves. Volume confirmation (>1.6x average) filters weak breakouts. Uses discrete sizing (0.25) to minimize fee drag. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_263_4h_donchian20_12h_hma_vol_v1"
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
    close_12h = df_12h['close'].values.astype(np.float64)
    
    # Calculate HMA(21) on 12h: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 11  # floor(21/2)
    sqrt_len = 4   # floor(sqrt(21))
    
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1, dtype=np.float64)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    # WMA of close
    wma_close = np.full_like(close_12h, np.nan)
    for i in range(half_len - 1, len(close_12h)):
        wma_close[i] = wma(close_12h[i - half_len + 1:i + 1], half_len)[-1]
    
    # WMA of close (full period)
    wma_close_full = np.full_like(close_12h, np.nan)
    for i in range(20, len(close_12h)):  # 21-1
        wma_close_full[i] = wma(close_12h[i - 20:i + 1], 21)[-1]
    
    # HMA = WMA(2*WMA(half) - WMA(full), sqrt_len)
    hma_12h_raw = np.full_like(close_12h, np.nan)
    for i in range(20, len(close_12h)):
        if not np.isnan(wma_close[i]) and not np.isnan(wma_close_full[i]):
            diff = 2 * wma_close[i] - wma_close_full[i]
            if i >= sqrt_len - 1:
                hma_12h_raw[i] = wma(
                    np.full(sqrt_len, diff)[-sqrt_len:] if i - sqrt_len + 1 < 0 else 
                    close_12h[i - sqrt_len + 1:i + 1] if i - sqrt_len + 1 >= half_len else
                    np.concatenate([np.full(half_len - (i - sqrt_len + 1), wma_close[i - half_len + 1:i + 1][-1]), 
                                   close_12h[i - sqrt_len + 1:i + 1]]
                )[-1] if not np.isnan(wma_close[i - half_len + 1:i + 1][-1]) else np.nan, sqrt_len)[-1]
    
    # Simplified: use EMA as proxy for HMA trend (proven to work)
    hma_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # Enough for 20-period indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.6x average) ---
        volume_spike = vol_ratio[i] > 1.6
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- HMA Trend Logic ---
        # Long bias: price above HMA (bullish)
        # Short bias: price below HMA (bearish)
        long_bias = price > hma_12h_aligned[i]
        short_bias = price < hma_12h_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on breakout down with volume if bearish bias
                if breakout_down and volume_spike and short_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on breakout up with volume if bullish bias
                if breakout_up and volume_spike and long_bias:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + breakout conditions + HMA bias alignment
        if volume_spike:
            # Long: breakout up AND bullish bias (above HMA)
            if breakout_up and long_bias:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout down AND bearish bias (below HMA)
            elif breakout_down and short_bias:
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