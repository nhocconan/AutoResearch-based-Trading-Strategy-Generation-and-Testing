#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # 1-hour data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4-hour trend filter (primary signal direction)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    sma_4h = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
    sma_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_4h)
    
    # 1-day volatility regime (filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # 1-hour ATR for entry timing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any data is not ready
        if (np.isnan(sma_4h_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        sma_4h_val = sma_4h_aligned[i]
        atr_1d = atr_1d_aligned[i]
        atr_ma_1d = atr_ma_1d_aligned[i]
        atr_val = atr[i]
        hour = hours[i]
        
        # Conditions: 4h trend + volatility regime + session
        uptrend = price > sma_4h_val
        downtrend = price < sma_4h_val
        vol_regime = atr_1d > atr_ma_1d
        in_session = 8 <= hour <= 20
        
        if position == 0 and vol_regime and in_session:
            # Long: uptrend + pullback to 4h SMA
            if uptrend and price <= sma_4h_val + 0.5 * atr_val:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + pullback to 4h SMA
            elif downtrend and price >= sma_4h_val - 0.5 * atr_val:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: opposite 4h trend or volatility collapse
            if (position == 1 and price < sma_4h_val) or \
               (position == -1 and price > sma_4h_val) or \
               atr_val < 0.3 * atr_1d:  # volatility collapse
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_4hTrend_Pullback_VolumeRegime_Session_v1"
timeframe = "1h"
leverage = 1.0