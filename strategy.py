#!/usr/bin/env python3
"""
Experiment #284: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian channel breakouts on daily timeframe capture strong momentum moves, while 1-week HMA ensures alignment with the primary trend. Volume confirmation filters out weak breakouts, and ATR-based stoploss manages risk. This structure works in both bull and bear markets by trading breakouts in the direction of the higher timeframe trend. Targets 20-50 trades over 4 years (5-12/year) to minimize fee drag while capturing high-probability trend continuation moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_2x_sub = 2 * wma_half - wma_full
        hma_21_1w = wma(wma_2x_sub, sqrt_len)
        
        # Pad beginning with NaN
        hma_21_1w_padded = np.full(len(close_1w), np.nan)
        hma_21_1w_padded[half_len + sqrt_len - 1:] = hma_21_1w[:len(hma_21_1w_padded) - (half_len + sqrt_len - 1)]
        
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w_padded)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel(20)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band + volume confirmation + price > 1w HMA
        long_breakout = close[i] > highest_high[i]
        long_volume = volume[i] > 1.5 * vol_avg_20[i]
        long_trend = close[i] > hma_21_1w_aligned[i]
        
        # Short: Price breaks below Donchian lower band + volume confirmation + price < 1w HMA
        short_breakout = close[i] < lowest_low[i]
        short_volume = volume[i] > 1.5 * vol_avg_20[i]
        short_trend = close[i] < hma_21_1w_aligned[i]
        
        if long_breakout and long_volume and long_trend:
            in_position = True
            position_side = 1
            entry_bar = i
            entry_price = close[i]
            signals[i] = SIZE
        elif short_breakout and short_volume and short_trend:
            in_position = True
            position_side = -1
            entry_bar = i
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals