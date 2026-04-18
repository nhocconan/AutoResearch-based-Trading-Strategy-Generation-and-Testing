#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and SMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate SMA(12) on daily close
    sma_12_1d = np.full(len(close_1d), np.nan)
    for i in range(12, len(close_1d)):
        sma_12_1d[i] = np.mean(close_1d[i-12:i])
    sma_12_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_12_1d)
    
    # Calculate volume moving average (12-period)
    vol_ma = np.full(n, np.nan)
    for i in range(12, n):
        vol_ma[i] = np.mean(volume[i-12:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 12)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(sma_12_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2 * 12-period average
        vol_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # Trend filter: price above daily SMA12 (uptrend) or below (downtrend)
        trend_up = close[i] > sma_12_1d_aligned[i]
        trend_down = close[i] < sma_12_1d_aligned[i]
        
        if position == 0:
            # Long entry: price above open + 0.5*ATR, volume and trend
            if (close[i] > open_price[i] + 0.5 * atr_1d_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price below open - 0.5*ATR, volume and trend
            elif (close[i] < open_price[i] - 0.5 * atr_1d_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below open or ATR-based stop
            if close[i] < open_price[i] - 1.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above open or ATR-based stop
            if close[i] > open_price[i] + 1.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ATR14Daily_SMA12Daily_VolumeFilter"
timeframe = "12h"
leverage = 1.0