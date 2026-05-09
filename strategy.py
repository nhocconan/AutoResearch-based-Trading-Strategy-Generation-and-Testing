#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate time-of-day filter once
    hours = prices.index.hour  # DatetimeIndex.hour works directly
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume average on 1d
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate Camarilla levels for 1h using previous bar's OHLC
    high_1 = prices['high'].values
    low_1 = prices['low'].values
    close_1 = prices['close'].values
    
    # Calculate pivot and ranges from previous bar
    pp = (high_1[:-1] + low_1[:-1] + close_1[:-1]) / 3
    r = high_1[:-1] - low_1[:-1]
    
    # Camarilla levels (R1, S1)
    r1 = pp + 1.1 * r / 12
    s1 = pp - 1.1 * r / 12
    
    # Shift levels to align with current bar (previous bar's levels)
    r1 = np.concatenate([[-999], r1[:-1]])
    s1 = np.concatenate([[999], s1[:-1]])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = vol_1d[i] > 1.5 * vol_ma20_1d_aligned[i]
        
        if position == 0:
            # Long: Close above R1 with 4h uptrend and volume spike
            if close_1[i] > r1[i] and close_1[i] > ema_50_4h_aligned[i] and vol_ok:
                signals[i] = 0.20
                position = 1
            # Short: Close below S1 with 4h downtrend and volume spike
            elif close_1[i] < s1[i] and close_1[i] < ema_50_4h_aligned[i] and vol_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Close below S1 or 4h trend turns down
            if close_1[i] < s1[i] or close_1[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Close above R1 or 4h trend turns up
            if close_1[i] > r1[i] or close_1[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals