#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R4_S4_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Camarilla R4 and S4 from previous 1d bar
    high_1d = high
    low_1d = low
    close_1d = close
    
    # Calculate daily range and Camarilla levels
    daily_range = high_1d - low_1d
    r4 = close_1d + daily_range * 1.500
    s4 = close_1d - daily_range * 1.500
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for EMA20
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_20_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close > R4 and price above 1w EMA20 with volume spike
            if close[i] > r4[i] and close[i] > ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < S4 and price below 1w EMA20 with volume spike
            elif close[i] < s4[i] and close[i] < ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < S4 or trend breaks (price < 1w EMA20)
            if close[i] < s4[i] or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > R4 or trend breaks (price > 1w EMA20)
            if close[i] > r4[i] or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals