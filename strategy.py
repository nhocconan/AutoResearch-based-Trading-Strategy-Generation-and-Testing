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
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend direction
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1_4h = df_4h['high'] - df_4h['low']
    tr2_4h = np.abs(df_4h['high'] - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3_4h = np.abs(df_4h['low'] - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Get 1d HTF data for session and higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for long-term trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session hours (08-20 UTC) to reduce noise
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_14_4h_aligned[i] > 0.003 * close[i]
        
        # Long conditions:
        # 1. 4h EMA(34) > 1d EMA(50) (bullish alignment across timeframes)
        # 2. Price > 4h EMA(34) (bullish momentum)
        # 3. Volatility filter (avoid chop)
        if (ema_34_4h_aligned[i] > ema_50_1d_aligned[i] and
            close[i] > ema_34_4h_aligned[i] and
            vol_filter):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 4h EMA(34) < 1d EMA(50) (bearish alignment across timeframes)
        # 2. Price < 4h EMA(34) (bearish momentum)
        # 3. Volatility filter (avoid chop)
        elif (ema_34_4h_aligned[i] < ema_50_1d_aligned[i] and
              close[i] < ema_34_4h_aligned[i] and
              vol_filter):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_EMA34_EMA50_Alignment_Vol_Filter_v1"
timeframe = "1h"
leverage = 1.0