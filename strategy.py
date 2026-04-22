#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volume filter
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in direction of weekly trend
        if ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]:  # Uptrend
            # Long: Price above weekly EMA + volume surge
            if (prices['close'].iloc[i] > ema_34_1w_aligned[i] and 
                prices['volume'].iloc[i] > 2.0 * vol_ma20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
        else:  # Downtrend
            # Short: Price below weekly EMA + volume surge
            if (prices['close'].iloc[i] < ema_34_1w_aligned[i] and 
                prices['volume'].iloc[i] > 2.0 * vol_ma20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        # Exit: Price crosses back below/above weekly EMA or volatility drops
        if position == 1:
            if (prices['close'].iloc[i] < ema_34_1w_aligned[i] or 
                atr_1d_aligned[i] < 0.3 * atr_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if (prices['close'].iloc[i] > ema_34_1w_aligned[i] or 
                atr_1d_aligned[i] < 0.3 * atr_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyTrend_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0