#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1h data for HTF (4h and 1d)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 10 or len(df_1d) < 5:
        return np.zeros(n)
    
    # 4h EMA50 for trend direction
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # 1d ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    # Session filter: 08-20 UTC (precomputed)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_4h_50_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_ok = vol_filter[i]
        sess_ok = session_filter[i]
        
        # Volatility filter: only trade when ATR > 0
        vol_filter_ok = atr_val > 0
        
        if position == 0:
            # Long: price above 4h EMA50 with volume and session
            if price > ema_trend and vol_ok and vol_filter_ok and sess_ok:
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA50 with volume and session
            elif price < ema_trend and vol_ok and vol_filter_ok and sess_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 4h EMA50 OR volatility drops OR session ends
            if price < ema_trend or not vol_filter_ok or not sess_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 4h EMA50 OR volatility drops OR session ends
            if price > ema_trend or not vol_filter_ok or not sess_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_EMA50_VolumeSession_Filter_v1"
timeframe = "1h"
leverage = 1.0