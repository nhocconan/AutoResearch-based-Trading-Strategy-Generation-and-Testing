#!/usr/bin/env python3
"""
Experiment #4716: 12h Donchian Breakout + 1d HMA Trend + Volume Filter
HYPOTHESIS: On 12h timeframe, price breaks above/below 20-period Donchian channels with volume confirmation and 1d HMA21 trend filter capture institutional breakouts in both bull and bear markets. The 12h timeframe minimizes fee drag while the Donchian structure provides clear entry/exit levels. Volume filter ensures breakout legitimacy, HMA21 on 1d filters counter-trend noise. Target: 12-37 trades/year to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4716_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: HMA21 for trend filter ===
    if len(df_1d) >= 21:
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = len(df_1d) // 2
        sqrt_len = int(np.sqrt(len(df_1d)))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full(len(arr), np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        
        close_1d = df_1d['close'].values
        wma_half = wma(close_1d, half_len) if half_len >= 1 else np.full(len(close_1d), np.nan)
        wma_full = wma(close_1d, len(close_1d)) if len(close_1d) >= 1 else np.full(len(close_1d), np.nan)
        
        # Pad to original length
        wma_half_padded = np.concatenate([np.full(len(close_1d)-len(wma_half), np.nan), wma_half]) if len(wma_half) > 0 else np.full(len(close_1d), np.nan)
        wma_full_padded = np.concatenate([np.full(len(close_1d)-len(wma_full), np.nan), wma_full]) if len(wma_full) > 0 else np.full(len(close_1d), np.nan)
        
        raw_hma = 2 * wma_half_padded - wma_full_padded
        hma_1d = wma(raw_hma, sqrt_len) if sqrt_len >= 1 else np.full(len(raw_hma), np.nan)
        hma_1d_padded = np.concatenate([np.full(len(raw_hma)-len(hma_1d), np.nan), hma_1d]) if len(hma_1d) > 0 else np.full(len(raw_hma), np.nan)
        
        # Align HTF HMA21 to 12h timeframe
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_padded)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 12h Indicators: Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(lookback, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Trend filter: price > HMA21(1d) for long, price < HMA21(1d) for short
        trend_filter_long = price > hma_1d_aligned[i]
        trend_filter_short = price < hma_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = (price >= highest_high[i]) and vol_confirm and trend_filter_long
        breakout_short = (price <= lowest_low[i]) and vol_confirm and trend_filter_short
        
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