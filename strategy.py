#!/usr/bin/env python3
"""
12h_KAMA_Direction_With_Volume_Confirmation_and_Chop_Filter
Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely, in ranging markets it stays flat.
Price above KAMA indicates uptrend, below indicates downtrend. Volume confirms institutional participation.
Choppy market filter (using Choppiness Index) prevents whipsaws in ranging conditions.
Designed for 12h timeframe to target 12-37 trades/year with minimal fee drag.
Works in both bull and bear markets via adaptive trend detection.
"""

name = "12h_KAMA_Direction_With_Volume_Confirmation_and_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close_array, er_length=10, fast_ma=2, slow_ma=30):
        change = np.abs(np.diff(close_array, n=er_length))
        volatility = np.sum(np.abs(np.diff(close_array)), axis=1)
        er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
        er = np.concatenate([np.full(er_length, np.nan), er])
        sc = (er * (2/(fast_ma+1) - 2/(slow_ma+1)) + 2/(slow_ma+1)) ** 2
        kama = np.full_like(close_array, np.nan)
        kama[er_length] = close_array[er_length]
        for i in range(er_length + 1, len(close_array)):
            if np.isnan(kama[i-1]) or np.isnan(sc[i]):
                kama[i] = close_array[i]
            else:
                kama[i] = kama[i-1] + sc[i] * (close_array[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    
    # Calculate Choppiness Index for regime filter
    def calculate_chop(high_array, low_array, close_array, period=14):
        atr = np.zeros_like(close_array)
        for i in range(1, len(close_array)):
            tr = max(high_array[i] - low_array[i],
                     np.abs(high_array[i] - close_array[i-1]),
                     np.abs(low_array[i] - close_array[i-1]))
            atr[i] = tr
        # Smooth ATR with Wilder's smoothing (alpha = 1/period)
        atr_smoothed = np.zeros_like(atr)
        atr_smoothed[period] = np.mean(atr[1:period+1])
        for i in range(period+1, len(atr)):
            atr_smoothed[i] = (atr_smoothed[i-1] * (period-1) + atr[i]) / period
        
        # Calculate highest high and lowest low over period
        highest_high = np.zeros_like(close_array)
        lowest_low = np.zeros_like(close_array)
        for i in range(period, len(close_array)):
            highest_high[i] = np.max(high_array[i-period+1:i+1])
            lowest_low[i] = np.min(low_array[i-period+1:i+1])
        
        # Chop calculation
        sum_atr = np.zeros_like(close_array)
        for i in range(period, len(close_array)):
            sum_atr[i] = np.sum(atr_smoothed[i-period+1:i+1])
        
        chop = np.full_like(close_array, 50.0)
        for i in range(period, len(close_array)):
            if highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(sum_atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Get 1d data for volume confirmation (using 1d volume average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.convolve(vol_1d, np.ones(20)/20, mode='same')
    vol_ma_1d[:10] = vol_ma_1d[20:30]  # Fill beginning with rolling values
    vol_ma_1d[-10:] = vol_ma_1d[-30:-20]  # Fill end with rolling values
    
    # Align 1d volume MA to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate current 12h volume ratio vs 1d average
    vol_ratio = np.divide(volume, vol_ma_1d_aligned, out=np.zeros_like(volume), where=vol_ma_1d_aligned!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Warmup for KAMA and CHOP
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending
        # For this strategy, we only trade in trending markets (chop < 38.2)
        trending_market = chop[i] < 38.2
        
        if not trending_market:
            # In ranging markets, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        if position == 0:
            # Enter long: price above KAMA in trending market with volume confirmation
            if price_above_kama and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA in trending market with volume confirmation
            elif price_below_kama and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price drops below KAMA or market becomes ranging
            if not price_above_kama or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above KAMA or market becomes ranging
            if not price_below_kama or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals