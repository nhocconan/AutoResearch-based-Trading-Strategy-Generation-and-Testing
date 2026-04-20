#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_200MA_Volume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly 200 EMA for trend
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 12h data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 2.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # start after warmup for weekly EMA200
        close_val = close[i]
        ema200_1w_val = ema200_1w_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(ema200_1w_val) or np.isnan(vol_spike_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly 200 EMA with volume spike
            if close_val > ema200_1w_val and vol_spike_val > 2.5:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly 200 EMA with volume spike
            elif close_val < ema200_1w_val and vol_spike_val > 2.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below weekly 200 EMA
            if close_val < ema200_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above weekly 200 EMA
            if close_val > ema200_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals