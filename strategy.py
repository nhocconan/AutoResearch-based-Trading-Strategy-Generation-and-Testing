#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover (21/50) with 4h trend filter and session filter (08-20 UTC)
# Long when 1h EMA(21) crosses above EMA(50) AND price > 4h EMA(50) AND within active session
# Short when 1h EMA(21) crosses below EMA(50) AND price < 4h EMA(50) AND within active session
# Exit when EMA(21) crosses back over EMA(50)
# Uses 4h EMA for trend filter to avoid counter-trend trades, session filter to reduce noise
# Target: 60-150 total trades over 4 years (15-37/year) for optimal 1h performance

name = "1h_ema21_50_4h_ema50_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # EMA(21) and EMA(50) on 1h
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # 4h EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if required data not available
        if np.isnan(ema_21[i]) or np.isnan(ema_50[i]) or np.isnan(ema_50_4h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Check exits: EMA(21) crosses back over EMA(50)
        if position == 1:  # long position
            if ema_21[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if ema_21[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with trend filter and session filter
            # Long: EMA(21) crosses above EMA(50) AND price > 4h EMA(50) AND in session
            if (ema_21[i] > ema_50[i] and ema_21[i-1] <= ema_50[i-1] and 
                close[i] > ema_50_4h_aligned[i] and in_session):
                signals[i] = 0.20
                position = 1
            # Short: EMA(21) crosses below EMA(50) AND price < 4h EMA(50) AND in session
            elif (ema_21[i] < ema_50[i] and ema_21[i-1] >= ema_50[i-1] and 
                  close[i] < ema_50_4h_aligned[i] and in_session):
                signals[i] = -0.20
                position = -1
    
    return signals