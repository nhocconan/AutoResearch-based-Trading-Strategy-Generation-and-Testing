#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    """
    1h Camarilla R1/S1 breakout with 4h trend filter and volume confirmation.
    - Long: Close breaks above R1 with volume > 1.5x avg and price > 4h EMA(34)
    - Short: Close breaks below S1 with volume > 1.5x avg and price < 4h EMA(34)
    - Exit: Close crosses back below R1 (long) or above S1 (short)
    - Uses Camarilla levels from previous hour (HLC of previous bar)
    - Session filter: 08-20 UTC
    - Target: 15-37 trades/year on 1h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate Camarilla levels for each hour using previous bar's HLC
    # R1 = Close + 1.1*(High - Low)/12
    # S1 = Close - 1.1*(High - Low)/12
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    camarilla_high = high_series.shift(1)  # previous hour high
    camarilla_low = low_series.shift(1)    # previous hour low
    camarilla_close = close_series.shift(1) # previous hour close
    camarilla_range = camarilla_high - camarilla_low
    r1 = camarilla_close + 1.1 * camarilla_range / 12
    s1 = camarilla_close - 1.1 * camarilla_range / 12
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if np.isnan(ema34_4h_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_ma20[i]) or not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Close breaks above R1 with volume confirmation and above 4h EMA trend
            if close[i] > r1[i] and vol_ok and close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Close breaks below S1 with volume confirmation and below 4h EMA trend
            elif close[i] < s1[i] and vol_ok and close[i] < ema34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Close crosses back below R1
            if close[i] < r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Close crosses back above S1
            if close[i] > s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals