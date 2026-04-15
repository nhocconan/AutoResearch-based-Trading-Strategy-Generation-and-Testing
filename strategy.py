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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / ma_20  # normalized bandwidth
    
    # Align 1d Bollinger Bands to 6h
    ma_20_6h = align_htf_to_ltf(prices, df_1d, ma_20)
    upper_bb_6h = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_6h = align_htf_to_ltf(prices, df_1d, lower_bb)
    bb_width_6h = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 6h - always true)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 6h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ma_20_6h[i]) or np.isnan(upper_bb_6h[i]) or 
            np.isnan(lower_bb_6h[i]) or np.isnan(bb_width_6h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Bollinger Band squeeze: low volatility environment
        # BB width < 5% indicates consolidation/squeeze
        is_squeeze = bb_width_6h[i] < 0.05
        
        # Volatility expansion: current ATR > 1.5x average ATR
        # We'll use a simple volatility expansion filter
        vol_expansion = atr_14[i] > 0.006 * close[i]  # ATR > 0.6% of price
        
        # Long conditions:
        # 1. Price breaks above upper Bollinger Band (breakout from squeeze)
        # 2. BB squeeze was present (low volatility environment)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility expansion: confirms breakout strength
        if (close[i] > upper_bb_6h[i] and
            is_squeeze and
            volume_ratio[i] > 1.5 and
            vol_expansion):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below lower Bollinger Band (breakdown from squeeze)
        # 2. BB squeeze was present (low volatility environment)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility expansion: confirms breakdown strength
        elif (close[i] < lower_bb_6h[i] and
              is_squeeze and
              volume_ratio[i] > 1.5 and
              vol_expansion):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_BollingerSqueeze_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0