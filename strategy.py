#!/usr/bin/env python3
"""
Experiment #4713: 4h Donchian Breakout + 12h HMA Trend + Volume Confirmation
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts aligned with 12h HMA(21) trend direction, confirmed by volume spikes (>1.5x 20-period average), capture institutional momentum with controlled trade frequency. Uses ATR(14) trailing stop (2.5x) for risk management. Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by 12h HMA).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4713_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for HMA(21) trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: HMA(21) for trend ===
    if len(df_12h) >= 21:
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 12 // 2
        sqrt_len = int(np.sqrt(12))
        
        def wma(arr, window):
            if len(arr) < window:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, window + 1, dtype=np.float64)
            return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
        
        close_12h = df_12h['close'].values
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 12)
        wma_diff = 2 * wma_half - wma_full
        hma_12h = wma(wma_diff, sqrt_len)
        
        # Pad beginning with NaN
        hma_12h_padded = np.full(len(close_12h), np.nan)
        if len(hma_12h) > 0:
            start_idx = len(close_12h) - len(hma_12h)
            hma_12h_padded[start_idx:] = hma_12h
        hma_12h = hma_12h_padded
    else:
        hma_12h = np.full(len(df_12h), np.nan)
    
    # Align HTF HMA to 4h timeframe
    if len(hma_12h) > 0:
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 4h Indicators: Volume confirmation ===
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
    
    warmup = max(lookback, 20, 14, 21)  # Donchian, Volume MA, ATR, HMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        breakout_long = (price >= highest_high[i]) and vol_confirm
        breakout_short = (price <= lowest_low[i]) and vol_confirm
        
        # Trend filter: 12h HMA direction
        # For long: price above 12h HMA (bullish bias)
        # For short: price below 12h HMA (bearish bias)
        if len(hma_12h_aligned) > i and not np.isnan(hma_12h_aligned[i]):
            hma_trend_long = price > hma_12h_aligned[i]
            hma_trend_short = price < hma_12h_aligned[i]
        else:
            hma_trend_long = True
            hma_trend_short = True
        
        # Final entry conditions: breakout + volume + trend alignment
        if breakout_long and vol_confirm and hma_trend_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short and vol_confirm and hma_trend_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals