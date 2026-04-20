#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 20-period EMA on daily close
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 20-period SMA for volume average
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # 14-period ATR on daily
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(vol_sma_20_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        atr = atr_14_1d_aligned[i]
        
        if position == 0:
            # Long: price above EMA20, volume spike, price near daily low (mean reversion in uptrend)
            if (price > ema_20_1d_aligned[i] and 
                vol > 2.0 * vol_sma_20_1d_aligned[i] and
                price < low_1d[i] + 0.5 * atr):  # within 0.5 ATR of daily low
                signals[i] = 0.25
                position = 1
            # Short: price below EMA20, volume spike, price near daily high (mean reversion in downtrend)
            elif (price < ema_20_1d_aligned[i] and 
                  vol > 2.0 * vol_sma_20_1d_aligned[i] and
                  price > high_1d[i] - 0.5 * atr):  # within 0.5 ATR of daily high
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA20 or reaches daily high (take profit)
            if price < ema_20_1d_aligned[i] or price >= high_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA20 or reaches daily low (take profit)
            if price > ema_20_1d_aligned[i] or price <= low_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA20_VolumeSpike_MeanReversion"
timeframe = "12h"
leverage = 1.0