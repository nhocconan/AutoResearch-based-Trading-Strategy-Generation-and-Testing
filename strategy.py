#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34) for trend
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Load daily data for Camarilla pivot levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volume average for confirmation
    vol_ma_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(close_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels from previous day's data
        if i == 0:
            continue
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        # Camarilla levels
        r4 = prev_close + range_ * 1.500
        r3 = prev_close + range_ * 1.250
        s3 = prev_close - range_ * 1.250
        s4 = prev_close - range_ * 1.500
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume and trend confirmation
            if (price > r4 and 
                price > ema_34_12h_aligned[i] and 
                vol > 1.5 * vol_ma_1d_aligned[i] and 
                atr_1d_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with volume and trend confirmation
            elif (price < s4 and 
                  price < ema_34_12h_aligned[i] and 
                  vol > 1.5 * vol_ma_1d_aligned[i] and 
                  atr_1d_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below R3 (mean reversion) or trend breaks
            if price < r3 or price < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above S3 (mean reversion) or trend breaks
            if price > s3 or price > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0