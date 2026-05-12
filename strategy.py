#!/usr/bin/env python3
name = "1d_Williams_Alligator_1wTrend_Threshold"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Williams Alligator on daily: Jaw (13), Teeth (8), Lips (5) SMMA
    smma13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    smma8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    smma5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Align 1w trend to daily
    sma20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    # Williams Fractals on daily for entry timing
    def williams_fractals(high_arr, low_arr):
        n = len(high_arr)
        bearish = np.full(n, np.nan)
        bullish = np.full(n, np.nan)
        for i in range(2, n-2):
            if (high_arr[i] > high_arr[i-1] and high_arr[i] > high_arr[i-2] and
                high_arr[i] > high_arr[i+1] and high_arr[i] > high_arr[i+2]):
                bearish[i] = high_arr[i]
            if (low_arr[i] < low_arr[i-1] and low_arr[i] < low_arr[i-2] and
                low_arr[i] < low_arr[i+1] and low_arr[i] < low_arr[i+2]):
                bullish[i] = low_arr[i]
        return bearish, bullish
    
    bearish_fractal, bullish_fractal = williams_fractals(high, low)
    # Fractals need 2-bar confirmation on daily
    bearish_fractal_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 20-day volume spike
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(smma13[i]) or np.isnan(smma8[i]) or np.isnan(smma5[i]) or
            np.isnan(sma20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Alligator aligned (Lips > Teeth > Jaw) + price above 1w sma20 + bullish fractal + volume
            if (smma5[i] > smma8[i] and smma8[i] > smma13[i] and
                close[i] > sma20_1w_aligned[i] and
                not np.isnan(bullish_fractal_aligned[i]) and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator inverted (Lips < Teeth < Jaw) + price below 1w sma20 + bearish fractal + volume
            elif (smma5[i] < smma8[i] and smma8[i] < smma13[i] and
                  close[i] < sma20_1w_aligned[i] and
                  not np.isnan(bearish_fractal_aligned[i]) and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Alligator misaligned OR price below 1w sma20
            if not (smma5[i] > smma8[i] and smma8[i] > smma13[i]) or close[i] < sma20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator misaligned OR price above 1w sma20
            if not (smma5[i] < smma8[i] and smma8[i] < smma13[i]) or close[i] > sma20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals