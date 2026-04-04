#!/usr/bin/env python3
"""
Experiment #5028: 12h Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation
HYPOTHESIS: On 12h timeframe, Donchian channel breakouts (20-bar) aligned with 1-week HMA(21) trend 
and volume confirmation (>1.5x average) capture high-probability swing trades. 
In bull markets: breakouts in direction of weekly trend. In bear markets: only take breakouts 
when price is above weekly HMA (avoiding false breakdowns). Designed for 12-37 trades/year on 12h 
timeframe to minimize fee drag while working in both bull (breakout continuation) and bear 
(selective long-only) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5028_12h_donchian20_1w_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: HMA(21) for trend filter ===
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        wma_half = np.full_like(close_1w, np.nan)
        wma_full = np.full_like(close_1w, np.nan)
        
        if len(close_1w) >= half_len:
            wma_half[half_len-1:] = wma(close_1w, half_len)
        if len(close_1w) >= 21:
            wma_full[20:] = wma(close_1w, 21)
        
        raw_hma = 2 * wma_half - wma_full
        hma_1w = np.full_like(raw_hma, np.nan)
        if len(raw_hma) >= sqrt_len:
            hma_1w[sqrt_len-1:] = wma(raw_hma[sqrt_len-1:], sqrt_len)
    else:
        hma_1w = np.full(len(df_1w), np.nan)
    
    # Align HTF HMA to 12h timeframe
    if len(hma_1w) > 0:
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # Precompute HTF: 1d data for Donchian channel calculation (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Donchian Channel (20) based on previous day ===
    if len(df_1d) >= 20:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Use previous day's data to avoid look-ahead
        high_prev = np.concatenate([[np.nan], high_1d[:-1]])
        low_prev = np.concatenate([[np.nan], low_1d[:-1]])
        
        # Calculate 20-period Donchian channels on previous day's data
        def rolling_max(arr, window):
            result = np.full_like(arr, np.nan)
            for i in range(window-1, len(arr)):
                result[i] = np.max(arr[i-window+1:i+1])
            return result
        
        def rolling_min(arr, window):
            result = np.full_like(arr, np.nan)
            for i in range(window-1, len(arr)):
                result[i] = np.min(arr[i-window+1:i+1])
            return result
        
        donch_high_prev = rolling_max(high_prev, 20)
        donch_low_prev = rolling_min(low_prev, 20)
    else:
        donch_high_prev = donch_low_prev = np.full(len(df_1d), np.nan)
    
    # Align HTF Donchian levels to 12h timeframe
    if len(donch_high_prev) > 0:
        donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_prev)
        donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_prev)
    else:
        donch_high_aligned = np.full(n, np.nan)
        donch_low_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Volume confirmation (1.5x spike) ===
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
    
    warmup = max(20, 20, 14)  # Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Determine market regime based on 1w HMA trend
        # Bullish: price > 1.01 * HMA (avoid whipsaws in strong uptrend)
        # Bearish: price < 0.99 * HMA (avoid false breakdowns in downtrend)
        bullish_regime = price > hma_1w_aligned[i] * 1.01
        bearish_regime = price < hma_1w_aligned[i] * 0.99
        
        # Long conditions: Donchian breakout above upper band
        # In bullish regime: take all breakouts
        # In bearish regime: only take breakouts if price is above HMA (avoid false signals)
        long_breakout = (price >= donch_high_aligned[i]) and vol_confirm and (bullish_regime or price > hma_1w_aligned[i])
        
        # Short conditions: Donchian breakdown below lower band
        # Only take shorts in bearish regime when price is below HMA
        short_breakout = (price <= donch_low_aligned[i]) and vol_confirm and bearish_regime and (price < hma_1w_aligned[i])
        
        # Final entry conditions
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals