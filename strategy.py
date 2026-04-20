#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 25:
        return np.zeros(n)
    
    # Load daily data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily ATR for volatility filter (14-period)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volume for confirmation (20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 12h ATR for position sizing adjustment (14-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    high_low_12h = high_12h - low_12h
    high_close_12h = np.abs(high_12h - np.roll(close_12h, 1))
    low_close_12h = np.abs(low_12h - np.roll(close_12h, 1))
    high_low_12h[0] = high_12h[0] - low_12h[0]
    high_close_12h[0] = np.abs(high_12h[0] - close_12h[0])
    low_close_12h[0] = np.abs(low_12h[0] - close_12h[0])
    tr_12h = np.maximum(high_low_12h, np.maximum(high_close_12h, low_close_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(25, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(close_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i] if 'volume_12h' in locals() else prices['volume'].values[i]
        atr = atr_12h[i]
        
        # Dynamic volatility filter: require ATR > 0.5 * 20-period average ATR
        vol_filter = atr > 0.5 * atr_1d_aligned[i]
        
        if position == 0:
            # Long: price above previous 12h high with volume confirmation and volatility filter
            if (i >= 1 and price > high_12h[i-1] and 
                vol > 1.5 * vol_ma_1d_aligned[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price below previous 12h low with volume confirmation and volatility filter
            elif (i >= 1 and price < low_12h[i-1] and 
                  vol > 1.5 * vol_ma_1d_aligned[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below previous 12h low or volatility drops
            if price < low_12h[i-1] or atr < 0.3 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above previous 12h high or volatility drops
            if price > high_12h[i-1] or atr < 0.3 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Breakout_VolumeVolatilityFilter"
timeframe = "12h"
leverage = 1.0