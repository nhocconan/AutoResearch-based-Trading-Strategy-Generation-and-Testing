#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Trix_Volume_Spike_v1"
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
    
    # Get 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate TRIX on 12h close: TRIX = EMA(EMA(EMA(close, 12), 12), 12) 
    # Then % change: (TRIX - TRIX_prev) / TRIX_prev * 100
    ema1 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = ema3.values
    trix = np.zeros_like(trix_raw)
    # Calculate percent change to avoid division by zero
    trix[1:] = (trix_raw[1:] - trix_raw[:-1]) / trix_raw[:-1] * 100
    trix[0] = 0
    
    # Align TRIX to 4h timeframe
    trix_4h = align_htf_to_ltf(prices, df_12h, trix)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 36  # Need enough history for TRIX calculation
    
    for i in range(start_idx, n):
        if np.isnan(trix_4h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.8x average
        volume_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike (bullish momentum)
            if trix_4h[i] > 0 and trix_4h[i-1] <= 0 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike (bearish momentum)
            elif trix_4h[i] < 0 and trix_4h[i-1] >= 0 and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below zero (momentum reversal)
            if trix_4h[i] < 0 and trix_4h[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above zero (momentum reversal)
            if trix_4h[i] > 0 and trix_4h[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals