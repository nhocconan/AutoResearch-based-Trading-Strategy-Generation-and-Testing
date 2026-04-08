#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_trend_following"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h trend: 20 EMA
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d trend: 50 EMA
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: 1h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume / vol_ma > 1.5
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # hold position
            else:
                signals[i] = 0.0
            continue
        
        # Only consider new signals during session with volume confirmation
        if not (in_session[i] and vol_confirm[i]):
            if position != 0:
                pass  # hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: close below 4h EMA
            if close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
                
        elif position == -1:  # Short position
            # Exit: close above 4h EMA
            if close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price above both 4h and 1d EMA with volume
            if (close[i] > ema_4h_aligned[i] and 
                close[i] > ema_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: price below both 4h and 1d EMA with volume
            elif (close[i] < ema_4h_aligned[i] and 
                  close[i] < ema_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals