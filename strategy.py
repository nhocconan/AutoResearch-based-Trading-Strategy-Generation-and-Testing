#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d_arr[0]], close_1d_arr[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d_arr[0]], close_1d_arr[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 6h ATR(14) for volatility filter (primary timeframe)
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_14_6h[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: avoid extremely low or high volatility
        vol_ratio_6h_to_1d = atr_14_6h[i] / (atr_14_1d_aligned[i] + 1e-10)
        vol_filter = (vol_ratio_6h_to_1d > 0.5) & (vol_ratio_6h_to_1d < 2.0)
        
        # Long conditions:
        # 1. 6h price above 1d EMA(34) (bullish bias)
        # 2. Volume confirmation: volume > 1.3x average
        # 3. Volatility regime: not too low, not too high
        if (close[i] > ema_34_1d_aligned[i] and
            volume_ratio[i] > 1.3 and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price below 1d EMA(34) (bearish bias)
        # 2. Volume confirmation: volume > 1.3x average
        # 3. Volatility regime: not too low, not too high
        elif (close[i] < ema_34_1d_aligned[i] and
              volume_ratio[i] > 1.3 and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA34_Volume_VolRegime_Filter"
timeframe = "6h"
leverage = 1.0