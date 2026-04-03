#!/usr/bin/env python3
"""
Experiment #390: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Daily Donchian channel breakouts filtered by weekly HMA trend and volume spike
capture strong momentum moves while avoiding whipsaws. The weekly trend filter ensures we
only trade in the direction of the higher timeframe momentum, reducing false breakouts.
Volume confirmation adds conviction to breakouts. Targets 20-50 trades over 4 years (5-12/year)
to minimize fee drag while maintaining statistical significance. Works in both bull and bear
markets by following the weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian20_1w_hma_vol_v1"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_2xhalf = np.concatenate([np.full(half_len, np.nan), wma_half]) if len(wma_half) > 0 else np.full(len(close_1w), np.nan)
        
        # Handle alignment
        raw_hma = 2 * wma_2xhalf - wma_full
        hma_1w = wma(raw_hma[~np.isnan(raw_hma)], sqrt_len) if len(raw_hma[~np.isnan(raw_hma)]) >= sqrt_len else np.array([])
        hma_1w_full = np.full(len(close_1w), np.nan)
        if len(hma_1w) > 0:
            start_idx = len(close_1w) - len(hma_1w)
            hma_1w_full[start_idx:] = hma_1w
        
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_full)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume average (Call ONCE before loop) ===
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === 1d Indicators: Donchian channels (20) ===
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    if n >= donchian_period:
        # Rolling max/min for Donchian channels
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        upper_channel = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
        lower_channel = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (Trailing stop based on ATR) ---
        if in_position:
            # Calculate ATR(14) for dynamic stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update highest high since entry
                highest_since_entry = max(highest_since_entry, high[i])
                # Trailing stop: exit if price drops 2.5*ATR from highest high
                stop_level = highest_since_entry - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below weekly HMA (trend change)
                if close[i] < hma_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Update lowest low since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Trailing stop: exit if price rises 2.5*ATR from lowest low
                stop_level = lowest_since_entry + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes above weekly HMA (trend change)
                if close[i] > hma_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: require volume spike (> 1.5x weekly average)
        volume_spike = vol_ratio_1w_aligned[i] > 1.5
        
        # Long: Price breaks above upper Donchian channel with volume and above weekly HMA
        long_condition = (
            close[i] > upper_channel[i] and 
            volume_spike and 
            close[i] > hma_1w_aligned[i]
        )
        
        # Short: Price breaks below lower Donchian channel with volume and below weekly HMA
        short_condition = (
            close[i] < lower_channel[i] and 
            volume_spike and 
            close[i] < hma_1w_aligned[i]
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals