#!/usr/bin/env python3
"""
Experiment #4894: 1h Donchian(20) Breakout + 4h/1d HMA Trend + Volume Spike
HYPOTHESIS: On 1h timeframe, Donchian(20) breakouts in direction of 4h HMA21 and 1d HMA50 trend with volume confirmation (>1.5x average) capture momentum moves. Uses 4h/1d for signal direction, 1h only for entry timing. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4894_1h_donchian20_4h_1d_hma_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Precompute HTF: 4h data for HMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    # Precompute HTF: 1d data for HMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: HMA21 for trend filter ===
    if len(df_4h) >= 21:
        # Hull Moving Average calculation
        half_len = len(df_4h) // 2
        sqrt_len = int(np.sqrt(len(df_4h)))
        
        # WMA function
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        close_4h = df_4h['close'].values
        wma_half = np.array([wma(close_4h[i:i+half_len], half_len)[-1] 
                            if i+half_len <= len(close_4h) else np.nan 
                            for i in range(len(close_4h))])
        wma_full = np.array([wma(close_4h[i:i+len(close_4h)], len(close_4h))[-1] 
                            if i+len(close_4h) <= len(close_4h) else np.nan 
                            for i in range(len(close_4h))])
        wma_sqrt = np.array([wma(close_4h[i:i+sqrt_len], sqrt_len)[-1] 
                            if i+sqrt_len <= len(close_4h) else np.nan 
                            for i in range(len(close_4h))])
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        hma_raw = 2 * wma_half - wma_full
        hma_4h = np.array([wma(hma_raw[i:i+sqrt_len], sqrt_len)[-1] 
                          if i+sqrt_len <= len(hma_raw) else np.nan 
                          for i in range(len(hma_raw))])
    else:
        hma_4h = np.full(len(df_4h), np.nan)
    
    # Align HTF HMA21 to 1h timeframe
    if len(hma_4h) > 0:
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    else:
        hma_4h_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: HMA50 for trend filter ===
    if len(df_1d) >= 50:
        # Hull Moving Average calculation
        half_len = len(df_1d) // 2
        sqrt_len = int(np.sqrt(len(df_1d)))
        
        # WMA function
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        close_1d = df_1d['close'].values
        wma_half = np.array([wma(close_1d[i:i+half_len], half_len)[-1] 
                            if i+half_len <= len(close_1d) else np.nan 
                            for i in range(len(close_1d))])
        wma_full = np.array([wma(close_1d[i:i+len(close_1d)], len(close_1d))[-1] 
                            if i+len(close_1d) <= len(close_1d) else np.nan 
                            for i in range(len(close_1d))])
        wma_sqrt = np.array([wma(close_1d[i:i+sqrt_len], sqrt_len)[-1] 
                            if i+sqrt_len <= len(close_1d) else np.nan 
                            for i in range(len(close_1d))])
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        hma_raw = 2 * wma_half - wma_full
        hma_1d = np.array([wma(hma_raw[i:i+sqrt_len], sqrt_len)[-1] 
                          if i+sqrt_len <= len(hma_raw) else np.nan 
                          for i in range(len(hma_raw))])
    else:
        hma_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF HMA50 to 1h timeframe
    if len(hma_1d) > 0:
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20)  # Donchian, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: reverse signal or stoploss via opposing breakout ---
        if in_position:
            # Check for opposing breakout to exit
            if position_side > 0:  # Long
                # Exit on Donchian lower break (stoploss) or opposing short signal
                if price <= low_roll[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit on Donchian upper break (stoploss) or opposing long signal
                if price >= high_roll[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with trend alignment (both 4h and 1d HMA)
        breakout_long = (price >= high_roll[i]) and \
                       (price > hma_4h_aligned[i]) and \
                       (price > hma_1d_aligned[i]) and \
                       vol_confirm
        breakout_short = (price <= low_roll[i]) and \
                        (price < hma_4h_aligned[i]) and \
                        (price < hma_1d_aligned[i]) and \
                        vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals