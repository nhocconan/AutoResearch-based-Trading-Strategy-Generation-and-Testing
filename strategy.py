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
    
    # Daily True Range and ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA(20)
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # 12h price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN
        if np.isnan(ema_20_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_20_val = ema_20_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_ma_val = volume_ma_20_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume must be above 20-day average
        vol_filter = vol > vol_ma_val
        
        if position == 0:
            # Long: price above EMA20, volume confirmation, and low volatility
            if price > ema_20_val and vol_filter and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 40):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA20, volume confirmation, and low volatility
            elif price < ema_20_val and vol_filter and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 40):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below EMA20 or volatility spikes
            if price < ema_20_val or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above EMA20 or volatility spikes
            if price > ema_20_val or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA20_VolumeVolatilityFilter"
timeframe = "12h"
leverage = 1.0