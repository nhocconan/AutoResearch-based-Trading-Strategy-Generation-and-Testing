#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsAlligator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d Williams Alligator: Jaw (13), Teeth (8), Lips (5)
    # Jaw: 13-period SMMA (smoothed moving average) - shifted by 8 bars
    close_1d = df_1d['close'].values
    sma_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    sma_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    sma_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Shift for Alligator: Jaw=8, Teeth=5, Lips=3
    jaw = np.roll(sma_13, 8)
    teeth = np.roll(sma_8, 5)
    lips = np.roll(sma_5, 3)
    
    # Fill NaN from roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d trend: EMA34 for filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator signals: Lips above Teeth above Jaw = uptrend
        # Lips below Teeth below Jaw = downtrend
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price > EMA34 + volume spike
            bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
            long_cond = bullish and (close[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            
            # Short: Lips < Teeth < Jaw (bearish alignment) + price < EMA34 + volume spike
            bearish = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
            short_cond = bearish and (close[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment (Lips < Teeth < Jaw) or price < EMA34
            bearish = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
            if bearish or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment (Lips > Teeth > Jaw) or price > EMA34
            bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
            if bullish or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals