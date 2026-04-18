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
    
    # Get 4H data for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4H EMA200 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema200_4h = close_4h_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4H EMA200 to 12H timeframe
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Calculate 12H Donchian(20) channels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Calculate 12H ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20, 14, 20)  # need EMA200, Donchian, ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * 20-period average ATR
        atr_ma = np.mean(tr[max(0, i-20):i]) if i >= 20 else np.nan
        vol_filter = not np.isnan(atr_ma) and atr[i] > 0.5 * atr_ma
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high, above 4H EMA200, with vol/vol filters
            if (close[i] > donch_high[i] and 
                close[i] > ema200_4h_aligned[i] and 
                vol_filter and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low, below 4H EMA200, with vol/vol filters
            elif (close[i] < donch_low[i] and 
                  close[i] < ema200_4h_aligned[i] and 
                  vol_filter and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below Donchian low or ATR drops too low
            if close[i] < donch_low[i] or (not np.isnan(atr_ma) and atr[i] < 0.3 * atr_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high or ATR drops too low
            if close[i] > donch_high[i] or (not np.isnan(atr_ma) and atr[i] < 0.3 * atr_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_4hEMA200_VolATR_Filter"
timeframe = "12h"
leverage = 1.0
EOF