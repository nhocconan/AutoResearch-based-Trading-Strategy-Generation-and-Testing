#!/usr/bin/env python3
"""
Experiment #5040: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts aligned with daily HMA trend capture strong momentum with controlled frequency. Daily HMA(21) acts as trend filter: only take longs when price > HMA, shorts when price < HMA. Volume > 1.5x average confirms participation. ATR(14) trailing stop (2.0x) manages risk. Designed for 19-50 trades/year on 4h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5040_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for HMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: HMA(21) for trend filter ===
    if len(df_1d) >= 21:
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half = len(df_1d) // 2
        sqrt_n = int(np.sqrt(len(df_1d)))
        
        def wma(arr, window):
            if len(arr) < window:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        
        close_1d = df_1d['close'].values
        wma_half = wma(close_1d, half) if half > 0 else np.array([])
        wma_full = wma(close_1d, len(close_1d))
        if len(wma_half) > 0 and len(wma_full) > 0:
            raw_hma = 2 * wma_half - wma_full
            hma_1d = wma(raw_hma, sqrt_n) if sqrt_n > 0 else raw_hma
            # Pad to original length
            hma_1d_padded = np.full(len(close_1d), np.nan)
            start_idx = len(close_1d) - len(hma_1d)
            if start_idx >= 0 and len(hma_1d) > 0:
                hma_1d_padded[start_idx:] = hma_1d
        else:
            hma_1d_padded = np.full(len(close_1d), np.nan)
        
        # Align to 4h timeframe
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_padded)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation (1.5x spike) ===
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Donchian breakout conditions with daily HMA trend filter
        # Long: Donchian breakout above AND price > daily HMA (uptrend)
        # Short: Donchian breakdown below AND price < daily HMA (downtrend)
        breakout_long = (price >= high_roll[i]) and (price > hma_1d_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < hma_1d_aligned[i]) and vol_confirm
        
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