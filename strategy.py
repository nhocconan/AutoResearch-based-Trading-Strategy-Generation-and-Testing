#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 10-day EMA for trend filter
    close_1d = df_1d['close'].values
    ema_10 = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - close_1d_prev)
    tr3 = np.abs(low_1d - close_1d_prev)
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Main timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_10_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_10_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_ok = vol_filter[i]
        
        # Volatility filter: only trade when ATR > 0.5% of price
        vol_filter_ok = atr_val > 0.005 * price
        
        if position == 0:
            # Long: price above EMA10 with volume and volatility
            if price > ema_val and vol_ok and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA10 with volume and volatility
            elif price < ema_val and vol_ok and vol_filter_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA10 or volatility drops
            if price < ema_val or not vol_filter_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA10 or volatility drops
            if price > ema_val or not vol_filter_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_EMA10_Trend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0