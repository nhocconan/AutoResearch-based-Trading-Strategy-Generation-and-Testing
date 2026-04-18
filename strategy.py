#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period SMA on 1d for trend filter
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d SMA to 4h
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Calculate 4h ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume spike (volume > 2.0x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate 4h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d SMA50
        uptrend = close[i] > sma_50_1d_aligned[i]
        downtrend = close[i] < sma_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]
        
        # Entry conditions: breakout of Donchian channel with trend and volume
        long_breakout = close[i] > donchian_high[i-1]
        short_breakout = close[i] < donchian_low[i-1]
        
        if position == 0:
            # Long: price breaks above Donchian high with uptrend and volume
            if long_breakout and uptrend and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with downtrend and volume
            elif short_breakout and downtrend and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian low or trend reverses
            if close[i] < donchian_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high or trend reverses
            if close[i] > donchian_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dSMA50_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0