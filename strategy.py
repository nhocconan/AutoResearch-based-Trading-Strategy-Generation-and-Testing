#!/usr/bin/env python3
"""
1h_4h1d_Momentum_Confluence
Uses 4h momentum (price > SMA50) and 1d trend (price > EMA200) for direction,
enters on 1h pullbacks to VWAP with volume confirmation.
Designed for low trade frequency (15-30/year) and works in both bull/bear markets.
"""

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
    
    # Get 4h data for momentum filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h SMA50 for momentum
    sma_50_4h = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
    sma_50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_50_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1h VWAP (typical price * volume) / volume
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_den = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = vwap_num / vwap_den
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 200  # need enough history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(sma_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vwap[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        sma_50_4h_val = sma_50_4h_aligned[i]
        ema_200_1d_val = ema_200_1d_aligned[i]
        vwap_val = vwap[i]
        vol_ma_val = vol_ma[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0 and in_session:
            # Long: 4h momentum up, 1d trend up, pullback to VWAP with volume
            if (price > sma_50_4h_val and 
                price > ema_200_1d_val and
                price <= vwap_val * 1.005 and  # within 0.5% above VWAP
                volume > vol_ma_val * 1.5):     # 1.5x average volume
                signals[i] = 0.20
                position = 1
            # Short: 4h momentum down, 1d trend down, bounce to VWAP with volume
            elif (price < sma_50_4h_val and 
                  price < ema_200_1d_val and
                  price >= vwap_val * 0.995 and  # within 0.5% below VWAP
                  volume > vol_ma_val * 1.5):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: momentum breaks down or price moves significantly above VWAP
            if price < sma_50_4h_val or price > vwap_val * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: momentum breaks up or price moves significantly below VWAP
            if price > sma_50_4h_val or price < vwap_val * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Momentum_Confluence"
timeframe = "1h"
leverage = 1.0