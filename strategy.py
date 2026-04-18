#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily data
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h EMA20 for trend
    close_series = pd.Series(close)
    ema20_4h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 14, 20)  # need EMA20, EMA50, ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema20_4h[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 20-period average ATR (avoid low volatility chop)
        atr_ma = np.mean(atr[i-20:i]) if i >= 20 else atr[i]
        vol_filter = atr[i] > 0.8 * atr_ma  # Require at least 80% of average volatility
        
        # Volume confirmation: current volume > 1.2 * 20-period average
        vol_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        if position == 0:
            # Long entry: price above daily EMA50, EMA20 trending up, with volatility and volume
            if (close[i] > ema50_1d_aligned[i] and 
                ema20_4h[i] > ema20_4h[i-1] and 
                vol_filter and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price below daily EMA50, EMA20 trending down, with volatility and volume
            elif (close[i] < ema50_1d_aligned[i] and 
                  ema20_4h[i] < ema20_4h[i-1] and 
                  vol_filter and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below EMA20 or volatility drops
            if close[i] < ema20_4h[i] or atr[i] < 0.5 * atr_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA20 or volatility drops
            if close[i] > ema20_4h[i] or atr[i] < 0.5 * atr_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA20_EMA50_Vol_Vol_Filter"
timeframe = "4h"
leverage = 1.0