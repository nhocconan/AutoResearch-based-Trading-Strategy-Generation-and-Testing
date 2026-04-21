# %%
#!/usr/bin/env python3
"""
12h_ChaikinMoneyFlow_TrendReversal
Hypothesis: Chaikin Money Flow (CMF) crossing zero on 12h timeframe with volume confirmation and 1d trend filter yields mean-reversion entries in both bull and bear markets. Uses 12h as primary timeframe with 1d HTF for trend confirmation. Targets 15-30 trades/year by requiring CMF zero-cross with volume spike and 1d EMA trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_cmf(high, low, close, volume, period=20):
    """Calculate Chaikin Money Flow"""
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # CMF = sum(MFV, period) / sum(volume, period)
    mfv_sum = np.zeros_like(mfv)
    vol_sum = np.zeros_like(volume)
    
    for i in range(len(mfv)):
        if i < period:
            mfv_sum[i] = np.sum(mfv[:i+1])
            vol_sum[i] = np.sum(volume[:i+1])
        else:
            mfv_sum[i] = np.sum(mfv[i-period+1:i+1])
            vol_sum[i] = np.sum(volume[i-period+1:i+1])
    
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)
    return cmf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate CMF on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    cmf = calculate_cmf(high, low, close, volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(cmf[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = volume[i-20:i].mean()
            volume_ok = vol > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price above/below 1d EMA
        price_above_ema = price > ema_1d_aligned[i]
        price_below_ema = price < ema_1d_aligned[i]
        
        # CMF signals: crossing zero
        cmf_cross_up = cmf[i] > 0 and cmf[i-1] <= 0
        cmf_cross_down = cmf[i] < 0 and cmf[i-1] >= 0
        
        if position == 0:
            # Long: CMF crosses above zero + volume + price above 1d EMA (bullish trend)
            if cmf_cross_up and volume_ok and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: CMF crosses below zero + volume + price below 1d EMA (bearish trend)
            elif cmf_cross_down and volume_ok and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CMF crosses below zero or volume drops
            if cmf[i] < 0 or not volume_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CMF crosses above zero or volume drops
            if cmf[i] > 0 or not volume_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ChaikinMoneyFlow_TrendReversal"
timeframe = "12h"
leverage = 1.0
# %%