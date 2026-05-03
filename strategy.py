#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (HMA21) and volume confirmation.
# Long when price breaks above upper Donchian channel in 1w uptrend (price > HMA21).
# Short when price breaks below lower Donchian channel in 1w downtrend (price < HMA21).
# Volume must be > 1.5x 20-period MA to confirm breakout strength.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 30-100 total trades over 4 years.
# Works in both bull and bear markets by filtering breakouts with higher-timeframe trend.

name = "1d_Donchian20_1wHMA21_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w HMA21
    close_1w = df_1w['close'].values
    half_len = int(21 / 2)
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_1w, half_len)
    wma_full = wma(close_1w, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_21_1w = wma(raw_hma, sqrt_len)
    
    # Pad beginning with NaN to align with original array
    hma_21_1w_padded = np.full(len(close_1w), np.nan)
    hma_21_1w_padded[half_len + sqrt_len - 1:] = hma_21_1w
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w_padded)
    
    # Donchian channels from 20-period high/low
    lookback = 20
    upper_chan = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_chan = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(hma_21_1w_aligned[i]) or np.isnan(upper_chan[i]) or 
            np.isnan(lower_chan[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > hma_21_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < hma_21_1w_aligned[i]  # 1w downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price breaks above upper Donchian AND 1w uptrend AND volume spike
            if close_val > upper_chan[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND 1w downtrend AND volume spike
            elif close_val < lower_chan[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR 1w trend turns down
            if close_val < lower_chan[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR 1w trend turns up
            if close_val > upper_chan[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals