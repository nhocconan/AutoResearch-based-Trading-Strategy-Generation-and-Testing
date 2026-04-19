#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_EMA200_Trend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 13-period EMA for Bull/Bear Power
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema200 = ema200_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Bull Power > 0, price above EMA200, volume confirmation
            if bull > 0 and price > ema200 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, price below EMA200, volume confirmation
            elif bear < 0 and price < ema200 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bull Power turns negative or price crosses below EMA200
            if bull <= 0 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bear Power turns positive or price crosses above EMA200
            if bear >= 0 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals