#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation (>1.5x average).
# Long when price breaks above upper Donchian channel in uptrend (close > HMA21) with volume spike.
# Short when price breaks below lower Donchian channel in downtrend (close < HMA21) with volume spike.
# Uses ATR-based trailing stop (2.0x ATR) to manage risk.
# Designed for low trade frequency (~20-50/year on 4h) to minimize fee drag while capturing strong directional moves.
# Works in bull markets via upper channel breakout continuation and in bear markets via lower channel breakdown continuation.
# Donchian levels from 12h provide institutional support/resistance that price respects.
# Uses discrete position sizing (0.25) to minimize churn and fee drag.

name = "4h_12hDonchian20_Breakout_12hHMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian calculations: highest high/lowest low over 20 periods
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 4h timeframe (wait for 12h bar to close)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Calculate 12h HMA(21) for trend filter
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def calculate_wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def calculate_hma(values, window):
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        wma_half = calculate_wma(values, half_window)
        wma_full = calculate_wma(values, window)
        # 2*WMA(n/2) - WMA(n)
        diff = 2 * wma_half - wma_full
        # Align arrays: wma_half starts at index half_window-1, wma_full at window-1
        # We need to align the diff array to the same length as values
        hma_raw = calculate_wma(diff, sqrt_window)
        # Pad with NaN to match original length
        hma = np.full_like(values, np.nan)
        # hma_raw starts at index (half_window-1) + (sqrt_window-1)
        start_idx = half_window - 1 + sqrt_window - 1
        end_idx = start_idx + len(hma_raw)
        if end_idx <= len(values) and start_idx >= 0:
            hma[start_idx:end_idx] = hma_raw
        return hma
    
    hma_21 = calculate_hma(close_12h, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # Calculate ATR(14) for dynamic trailing stop on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    start_idx = 50  # warmup for Donchian and HMA calculation
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        elif i > 0:
            vol_ma_20 = np.mean(volume[:i])
        else:
            vol_ma_20 = 0
        volume_spike = volume[i] > (1.5 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_upper = upper_aligned[i]
        curr_lower = lower_aligned[i]
        curr_hma = hma_21_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if volume_spike and not np.isnan(curr_upper) and not np.isnan(curr_lower) and not np.isnan(curr_hma):
                # Breakout longs: price breaks above upper Donchian in uptrend
                if curr_close > curr_upper and curr_close > curr_hma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Breakout shorts: price breaks below lower Donchian in downtrend
                elif curr_close < curr_lower and curr_close < curr_hma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals