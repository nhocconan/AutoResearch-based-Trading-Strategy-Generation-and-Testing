#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels on 1h close
    pivot = (high + low + close) / 3
    range_val = high - low
    r1 = pivot + 1.1 * range_val / 12
    s1 = pivot - 1.1 * range_val / 12
    
    # 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d volume filter: volume > 1.5 * 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(r1[i]) or np.isnan(s1[i]) or \
           np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume spike + session
            if (price > r1[i] and 
                price > ema50_4h_aligned[i] and 
                vol_filter[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
                continue
            
            # Short: price breaks below S1 + 4h downtrend + volume spike + session
            elif (price < s1[i] and 
                  price < ema50_4h_aligned[i] and 
                  vol_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price retreats below pivot or 4h trend fails
            if (price < pivot[i] or 
                price < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises above pivot or 4h trend fails
            if (price > pivot[i] or 
                price > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals