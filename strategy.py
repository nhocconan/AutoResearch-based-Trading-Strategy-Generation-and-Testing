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
    
    # Get weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ATR to 6h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 6h ATR for position sizing
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr1_6h[0] = 0
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Hour filter: 0-6 UTC (Asian session - lower volatility)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1w_aligned[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 0-6 UTC (Asian session)
        hour = hours[i]
        in_session = 0 <= hour <= 6
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: weekly ATR below median (low volatility environment)
        vol_filter = atr_1w_aligned[i] < np.nanmedian(atr_1w_aligned[max(0, i-100):i+1])
        
        # Entry conditions: 
        # Long: price > 6h EMA(20) and weekly ATR low
        # Short: price < 6h EMA(20) and weekly ATR low
        ema_20 = pd.Series(close).ewm(span=20, min_periods=20).mean().values
        
        long_entry = (close[i] > ema_20[i]) and vol_filter
        short_entry = (close[i] < ema_20[i]) and vol_filter
        
        # Exit conditions: price crosses EMA(20) in opposite direction
        long_exit = (close[i] < ema_20[i])
        short_exit = (close[i] > ema_20[i])
        
        # Position sizing: 0.25 for normal volatility, 0.15 for high volatility
        if vol_filter:
            size = 0.25
        else:
            size = 0.15
        
        if long_entry and position <= 0:
            signals[i] = size
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = size
            elif position == -1:
                signals[i] = -size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_AsianSession_EMA20_WeeklyATR_Filter"
timeframe = "6h"
leverage = 1.0