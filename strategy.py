#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour KAMA trend direction with 1-day Williams Alligator filter
# KAMA adapts to market noise, reducing whipsaw in chop; Williams Alligator confirms trend alignment.
# Designed for 4h timeframe to achieve 20-50 trades/year with low decay.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h KAMA (ER=10) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Efficiency Ratio
    change = np.abs(close_4h - np.concatenate([[close_4h[0]], close_4h[:-10]]))
    erosion = np.sum(np.abs(np.diff(close_4h)), axis=0) if len(close_4h) >= 10 else np.full_like(close_4h, np.nan)
    # Simplified erosion calculation for rolling window
    erosion_roll = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if i < 10:
            erosion_roll[i] = np.nan
        else:
            erosion_roll[i] = np.sum(np.abs(np.diff(close_4h[i-9:i+1])))
    er = change / np.where(erosion_roll == 0, 1, erosion_roll)
    er = np.where(np.isnan(er), 0, er)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    kama_slope = kama - np.concatenate([[kama[0]], kama[:-1]])
    kama_up = kama_slope > 0
    kama_down = kama_slope < 0
    
    kama_up_aligned = align_htf_to_ltf(prices, df_4h, kama_up)
    kama_down_aligned = align_htf_to_ltf(prices, df_4h, kama_down)
    
    # === 1-day Williams Alligator ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_price = (high_1d + low_1d + close_1d) / 3
    
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw = np.zeros_like(median_price)
    for i in range(len(median_price)):
        if i < 13:
            jaw[i] = np.nan
        else:
            jaw[i] = np.mean(median_price[i-12:i+1])
    jaw_shifted = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth = np.zeros_like(median_price)
    for i in range(len(median_price)):
        if i < 8:
            teeth[i] = np.nan
        else:
            teeth[i] = np.mean(median_price[i-7:i+1])
    teeth_shifted = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    
    # Lips (5-period SMMA, 3 bars ahead)
    lips = np.zeros_like(median_price)
    for i in range(len(median_price)):
        if i < 5:
            lips[i] = np.nan
        else:
            lips[i] = np.mean(median_price[i-4:i+1])
    lips_shifted = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Alligator alignment: Lips > Teeth > Jaw = up, Lips < Teeth < Jaw = down
    alligator_up = (lips_shifted > teeth_shifted) & (teeth_shifted > jaw_shifted)
    alligator_down = (lips_shifted < teeth_shifted) & (teeth_shifted < jaw_shifted)
    
    alligator_up_aligned = align_htf_to_ltf(prices, df_1d, alligator_up)
    alligator_down_aligned = align_htf_to_ltf(prices, df_1d, alligator_down)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_up_aligned[i]) or np.isnan(kama_down_aligned[i]) or
            np.isnan(alligator_up_aligned[i]) or np.isnan(alligator_down_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            if kama_up_aligned[i] and alligator_up_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            elif kama_down_aligned[i] and alligator_down_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: exit when trend alignment breaks
        elif position == 1:
            if not (kama_up_aligned[i] and alligator_up_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if not (kama_down_aligned[i] and alligator_down_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Alligator_Trend_Filter"
timeframe = "4h"
leverage = 1.0