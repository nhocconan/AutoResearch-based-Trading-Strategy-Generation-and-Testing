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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily ATR(50) for long-term volatility (regime filter)
    atr_50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate 6h ATR(14) for volatility entry filter
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_50_1d_aligned[i]) or 
            np.isnan(atr_14_6h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated but not extreme
        # Avoid both low-volatility chop and extreme volatility spikes
        vol_ratio = atr_14_1d_aligned[i] / (atr_50_1d_aligned[i] + 1e-10)
        vol_regime = (vol_ratio > 0.8) & (vol_ratio < 2.0)
        
        # Calculate dynamic thresholds based on volatility
        # Higher volatility = higher breakout threshold needed
        vol_scalar = np.clip(vol_ratio, 0.5, 2.0)
        breakout_threshold = 0.006 * vol_scalar  # Base 0.6% scaled by volatility
        
        # Long conditions:
        # 1. Price breaks above recent high with volume (bullish continuation)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. 6h ATR sufficient for move
        # 4. Volatility regime filter (avoid chop and extreme vol)
        if (close[i] > close[i-1] * (1 + breakout_threshold) and
            volume_ratio[i] > 1.5 and
            atr_14_6h[i] > 0.003 * close[i] and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below recent low with volume (bearish continuation)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. 6h ATR sufficient for move
        # 4. Volatility regime filter
        elif (close[i] < close[i-1] * (1 - breakout_threshold) and
              volume_ratio[i] > 1.5 and
              atr_14_6h[i] > 0.003 * close[i] and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Vol_Regime_Dynamic_Breakout_v1"
timeframe = "6h"
leverage = 1.0