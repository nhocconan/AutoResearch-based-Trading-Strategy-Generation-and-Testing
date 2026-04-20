#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA (34-period) for trend
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily volume average (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 12h price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma_20, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in aligned data
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_ma_20_val = vol_ma_20_1d_aligned[i]
        price = close[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: price above daily EMA34 + volume expansion + low volatility
            if (price > ema_34_val and 
                vol_ratio_val > 1.5 and 
                atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 50)):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34 + volume expansion + low volatility
            elif (price < ema_34_val and 
                  vol_ratio_val > 1.5 and 
                  atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 50)):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below EMA34 or volatility contraction
            if price < ema_34_val or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above EMA34 or volatility contraction
            if price > ema_34_val or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA34_VolumeExpansion_VolatilityFilter"
timeframe = "12h"
leverage = 1.0