#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Trend_Follow_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (weekly EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA21 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume spike filter: current volume > 1.5 * 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 30)  # Need enough data for EMA21 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema21 = ema21_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close above weekly EMA + volume spike
            if close[i] > ema21 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close below weekly EMA + volume spike
            elif close[i] < ema21 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below weekly EMA
            if close[i] < ema21:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above weekly EMA
            if close[i] > ema21:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals