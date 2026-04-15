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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d HTF data for ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1h ATR(14) for stoploss
    tr1_1h = high - low
    tr2_1h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_1h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_1h = np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))
    atr_14_1h = pd.Series(tr_1h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_14_1h[i]) or 
            np.isnan(volume_ratio[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 1h price above 4h EMA20 (bullish trend)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: 1d ATR > 0.8% of price (avoid low volatility)
        if (close[i] > ema_20_4h_aligned[i] and
            volume_ratio[i] > 1.5 and
            atr_14_1d_aligned[i] > 0.008 * close[i]):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 1h price below 4h EMA20 (bearish trend)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: 1d ATR > 0.8% of price
        elif (close[i] < ema_20_4h_aligned[i] and
              volume_ratio[i] > 1.5 and
              atr_14_1d_aligned[i] > 0.008 * close[i]):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_EMA20_1d_ATR_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0