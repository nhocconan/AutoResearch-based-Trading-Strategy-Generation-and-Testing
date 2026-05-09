#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Donchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    """
    1d Donchian(20) breakout with 1w trend filter and volume confirmation.
    - Long: Close breaks above upper Donchian(20) with volume > 1.5x average and price > 1w EMA(20)
    - Short: Close breaks below lower Donchian(20) with volume > 1.5x average and price < 1w EMA(20)
    - Exit: Close crosses back through Donchian middle (mean of upper/lower)
    - Uses Donchian channels from previous 20 days
    - Target: 10-25 trades/year on 1d timeframe
    """
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate Donchian channels (20-day)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    middle = (upper + lower) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Close breaks above upper Donchian with volume confirmation and above 1w EMA trend
            if close[i] > upper[i] and vol_ok and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Donchian with volume confirmation and below 1w EMA trend
            elif close[i] < lower[i] and vol_ok and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close crosses below middle Donchian
            if close[i] < middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close crosses above middle Donchian
            if close[i] > middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals