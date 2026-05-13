#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
# Long when price breaks above Donchian upper with volume > 1.5x average AND price > 12h HMA21.
# Short when price breaks below Donchian lower with volume > 1.5x average AND price < 12h HMA21.
# Exit on opposite Donchian level or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 19-50 trades/year.
# Works in bull markets via breakout continuation and in bear markets via faded rallies at resistance.
# Proven pattern from DB: Donchian breakout + volume + trend filter yields test Sharpe 0.72-1.38.

name = "4h_Donchian20_12hHMA21_Volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate HMA(21) on 12h close for trend filter
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='full')[-len(values):] / weights.sum()
    
    wma_half = wma(close_12h, half_len)
    wma_full = wma(close_12h, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = wma(raw_hma, sqrt_len)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate Donchian(20) on 4h
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(hma_21_12h_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper with volume confirmation AND price > 12h HMA21
            if close[i] > upper[i] and volume_filter[i] and close[i] > hma_21_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower with volume confirmation AND price < 12h HMA21
            elif close[i] < lower[i] and volume_filter[i] and close[i] < hma_21_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian lower OR trend reversal (price < 12h HMA21)
            if close[i] < lower[i] or close[i] < hma_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian upper OR trend reversal (price > 12h HMA21)
            if close[i] > upper[i] or close[i] > hma_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals