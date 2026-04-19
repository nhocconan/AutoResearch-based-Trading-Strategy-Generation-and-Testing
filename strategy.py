#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_EMA200_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 and Elder Ray calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily closes
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Bull Power and Bear Power on daily data
    bull_power_1d = high_1d - ema200_1d
    bear_power_1d = low_1d - ema200_1d
    
    # Align Elder Ray components to 6h timeframe
    ema200_6h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 60-period EMA on 6h close for trend filter
    ema60_6h = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_6h[i]) or np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or \
           np.isnan(ema60_6h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema200 = ema200_6h[i]
        bull_power = bull_power_6h[i]
        bear_power = bear_power_6h[i]
        ema60 = ema60_6h[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying pressure) + price above EMA60 + volume
            if bull_power > 0 and price > ema60 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling pressure) + price below EMA60 + volume
            elif bear_power < 0 and price < ema60 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bull Power turns negative or price drops below EMA200
            if bull_power <= 0 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bear Power turns positive or price rises above EMA200
            if bear_power >= 0 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals