#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA(34) - trend filter
    close_1w = df_1w['close'].values
    ema_1w_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR(14) for volatility filter and stop
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily ATR(14) for stop loss calculation
    atr_stop_multiplier = 2.5
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    
    # Warmup
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_34_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above weekly EMA with volatility expansion
            if (price > ema_1w_34_aligned[i] and 
                atr_1d_aligned[i] > atr_1d_aligned[i-1] * 1.2):  # volatility increasing
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below weekly EMA with volatility expansion
            elif (price < ema_1w_34_aligned[i] and 
                  atr_1d_aligned[i] > atr_1d_aligned[i-1] * 1.2):  # volatility increasing
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: ATR-based stop or trend reversal
            if (price <= entry_price - atr_stop_multiplier * atr_1d_aligned[i] or
                price < ema_1w_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ATR-based stop or trend reversal
            if (price >= entry_price + atr_stop_multiplier * atr_1d_aligned[i] or
                price > ema_1w_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA34_VolatilityBreakout_ATRStop_v1"
timeframe = "1d"
leverage = 1.0