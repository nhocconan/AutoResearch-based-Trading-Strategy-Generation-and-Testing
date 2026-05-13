#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel with volume > 1.3x average AND price > 12h HMA21.
# Short when price breaks below lower Donchian channel with volume > 1.3x average AND price < 12h HMA21.
# Exit on opposite Donchian level (lower for longs, upper for shorts) or trend reversal.
# Uses discrete position sizing (0.30) to balance performance and fee drag. Target: 20-50 trades/year.
# Works in bull markets via breakout continuation and in bear markets via faded rallies at resistance.

name = "4h_Donchian20_12hHMA21_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # Pad arrays to match original length
    wma_half_padded = np.full_like(close, np.nan)
    wma_full_padded = np.full_like(close, np.nan)
    wma_half_padded[half_period-1:] = wma_half
    wma_full_padded[period-1:] = wma_full
    
    # Calculate raw HMA
    raw_hma = 2 * wma_half_padded - wma_full_padded
    hma = wma(raw_hma, sqrt_period)
    
    # Pad final result
    hma_padded = np.full_like(close, np.nan)
    hma_padded[sqrt_period-1:] = hma[:-sqrt_period+1] if len(hma) >= sqrt_period else hma
    return hma_padded

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h close for trend filter
    hma21_12h = calculate_hma(close_12h, 21)
    hma21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma21_12h)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    # Upper channel = highest high over past 20 periods
    # Lower channel = lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(hma21_12h_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or \
           np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian with volume confirmation AND price > 12h HMA21
            if close[i] > upper_channel[i] and volume_filter[i] and close[i] > hma21_12h_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: price breaks below lower Donchian with volume confirmation AND price < 12h HMA21
            elif close[i] < lower_channel[i] and volume_filter[i] and close[i] < hma21_12h_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below lower Donchian OR trend reversal (price < 12h HMA21)
            if close[i] < lower_channel[i] or close[i] < hma21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price breaks above upper Donchian OR trend reversal (price > 12h HMA21)
            if close[i] > upper_channel[i] or close[i] > hma21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals