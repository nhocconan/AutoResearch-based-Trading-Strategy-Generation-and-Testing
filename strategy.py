#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE (1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period ATR on 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 10-day SMA volume
    sma_vol_10d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1d EMA200
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 12h (wait for daily bar close)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    sma_vol_10d_aligned = align_htf_to_ltf(prices, df_1d, sma_vol_10d)
    
    # Calculate 12h price array
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN
        if np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(sma_vol_10d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_200_val = ema_200_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_sma = sma_vol_10d_aligned[i]
        price = close[i]
        vol_current = volume_1d[i // 288] if i >= 288 else volume_1d[0]  # approximate 12h volume from daily
        
        if position == 0:
            # Long: price above EMA200, volume above average, and volatility is low
            if price > ema_200_val and vol_current > vol_sma and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 30):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA200, volume above average, and volatility is low
            elif price < ema_200_val and vol_current > vol_sma and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 30):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below EMA200 or volatility spikes above 50th percentile
            if price < ema_200_val or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above EMA200 or volatility spikes above 50th percentile
            if price > ema_200_val or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA200_Volume_VolatilityFilter"
timeframe = "12h"
leverage = 1.0