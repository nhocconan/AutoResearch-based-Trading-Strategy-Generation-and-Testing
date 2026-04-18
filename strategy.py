#!/usr/bin/env python3
"""
6h Time-Based Momentum with Volume Confirmation and 12h Trend Filter
Enters long/short at specific UTC hours when momentum typically accelerates,
filtered by 12h EMA trend and volume spikes. Designed to work in both bull
and bear markets by capturing institutional session momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection (1.5x 24-period average - 6 days worth)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute hour filter (UTC 8-16 = London/NY overlap)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 16)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: in session, price above 12h EMA, volume spike
            if (in_session[i] and 
                price > ema_trend and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: in session, price below 12h EMA, volume spike
            elif (in_session[i] and 
                  price < ema_trend and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 12h EMA or outside session
            if price < ema_trend or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12h EMA or outside session
            if price > ema_trend or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_SessionMomentum_Volume_12hEMA"
timeframe = "6h"
leverage = 1.0