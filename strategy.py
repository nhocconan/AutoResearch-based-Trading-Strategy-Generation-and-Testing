#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Channel_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Keltner Channel and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(20) as center line
    ema20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d ATR(10) for channel width
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    low_close = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # First TR
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Keltner Channel bands: ±1.5 * ATR
    upper_keltner = ema20_1d + 1.5 * atr10_1d
    lower_keltner = ema20_1d - 1.5 * atr10_1d
    
    # Align Keltner Channel levels to 4h
    upper_keltner_4h = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_4h = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema20_1d_4h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume spike detection (20-period for 4h)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_keltner_4h[i]) or np.isnan(lower_keltner_4h[i]) or 
            np.isnan(ema20_1d_4h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above Upper Keltner with uptrend and volume spike
            if close[i] > upper_keltner_4h[i] and close[i] > ema20_1d_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Lower Keltner with downtrend and volume spike
            elif close[i] < lower_keltner_4h[i] and close[i] < ema20_1d_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below EMA(20) OR trend turns down
            if close[i] < ema20_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above EMA(20) OR trend turns up
            if close[i] > ema20_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals