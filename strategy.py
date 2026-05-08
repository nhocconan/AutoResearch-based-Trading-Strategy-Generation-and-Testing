#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Momentum_Retest_1dVWAP_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily VWAP
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * df_1d['volume'].values)
    vwap_denominator = np.cumsum(df_1d['volume'].values)
    vwap_1d = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, 0.0)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # 6h momentum - rate of change over 3 periods (18 hours)
    roc = np.zeros(n)
    roc[3:] = (close[3:] - close[:-3]) / close[:-3]
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(roc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above VWAP, above EMA50, positive momentum, volume confirmation
            if (close[i] > vwap_1d_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and
                roc[i] > 0.005 and  # 0.5% momentum threshold
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP, below EMA50, negative momentum, volume confirmation
            elif (close[i] < vwap_1d_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  roc[i] < -0.005 and  # -0.5% momentum threshold
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below VWAP OR below EMA50
            if close[i] < vwap_1d_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above VWAP OR above EMA50
            if close[i] > vwap_1d_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals