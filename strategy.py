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
    
    # Calculate daily Williams %R(14) for oversold/overbought conditions
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max().values
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12h ATR(14) for volatility entry filter
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_12h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_14_12h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.6% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.006 * close[i]
        
        # Long conditions:
        # 1. Williams %R below -80 (oversold)
        # 2. Volume confirmation: volume > 1.4x average
        # 3. 12h ATR > 0.3% of price (ensure sufficient volatility for move)
        # 4. Daily volatility regime filter (avoid chop)
        if (williams_r_aligned[i] < -80 and
            volume_ratio[i] > 1.4 and
            atr_14_12h[i] > 0.003 * close[i] and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Williams %R above -20 (overbought)
        # 2. Volume confirmation: volume > 1.4x average
        # 3. 12h ATR > 0.3% of price
        # 4. Daily volatility regime filter
        elif (williams_r_aligned[i] > -20 and
              volume_ratio[i] > 1.4 and
              atr_14_12h[i] > 0.003 * close[i] and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_Volume_Volatility_Regime_v1"
timeframe = "12h"
leverage = 1.0