#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 21-period EMA on weekly for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMA to 12h
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 12h ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume spike (volume > 2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA21
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]
        
        if position == 0:
            # Long: price above weekly EMA21 with volume spike
            if uptrend and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA21 with volume spike
            elif downtrend and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA21
            if close[i] < ema_21_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA21
            if close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_weeklyEMA21_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0