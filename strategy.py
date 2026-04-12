#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elliott_wave_oscillator_v1"
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
    
    # Daily data for Elliott Wave Oscillator (EWO) and volume profile
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 5-period and 34-period SMAs for EWO
    sma5_1d = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().values
    sma34_1d = pd.Series(df_1d['close']).rolling(window=34, min_periods=34).mean().values
    ewo_1d = sma5_1d - sma34_1d  # Elliott Wave Oscillator
    
    # Align EWO to 6h timeframe
    ewo_aligned = align_htf_to_ltf(prices, df_1d, ewo_1d)
    
    # Daily volume for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Volume moving average (20-period)
    volume_ma = pd.Series(volume_aligned).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ewo_aligned[i]) or np.isnan(volume_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 20-period average
        volume_filter = volume[i] > volume_ma[i]
        
        # EWO signals: positive = bullish momentum, negative = bearish momentum
        ewo_long = ewo_aligned[i] > 0 and volume_filter
        ewo_short = ewo_aligned[i] < 0 and volume_filter
        
        # Exit when EWO crosses zero (momentum shift)
        exit_long = position == 1 and ewo_aligned[i] <= 0
        exit_short = position == -1 and ewo_aligned[i] >= 0
        
        # Execute trades
        if ewo_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif ewo_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals