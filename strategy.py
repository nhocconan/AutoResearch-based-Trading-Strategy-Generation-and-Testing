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
    
    # Get 1d data for calculations (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR(20) for volatility calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 6h ATR(20) for volatility ratio
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=20, min_periods=20).mean().values
    
    # Volatility ratio: 6h ATR / 1d ATR (normalized)
    atr_ratio = atr_6h / atr_1d
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Trend filter: 6h close > 6h EMA34 (trending up) or < EMA34 (trending down)
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema34[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: volatility contraction (ratio < 0.8) + volume expansion + uptrend
            if (atr_ratio[i] < 0.8 and 
                volume_filter[i] and 
                close[i] > ema34[i]):
                signals[i] = 0.25
                position = 1
            # Short: volatility contraction + volume expansion + downtrend
            elif (atr_ratio[i] < 0.8 and 
                  volume_filter[i] and 
                  close[i] < ema34[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: volatility expansion (ratio > 1.2) or trend reversal
            if (atr_ratio[i] > 1.2 or 
                close[i] < ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: volatility expansion or trend reversal
            if (atr_ratio[i] > 1.2 or 
                close[i] > ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolRatio_VolumeTrend_Breakout_v1"
timeframe = "6h"
leverage = 1.0