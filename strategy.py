#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    """
    1d Camarilla R1/S1 breakout with 1w trend filter and volume confirmation.
    - Long: Close breaks above R1 with volume > 2x avg and price > 1w EMA(20)
    - Short: Close breaks below S1 with volume > 2x avg and price < 1w EMA(20)
    - Exit: Close crosses back through Camarilla pivot (P) level
    - Uses Camarilla from previous day (excluding current)
    - Target: 10-25 trades/year on 1d timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate Camarilla levels from previous day (using HLC of previous day)
    # Camarilla: P = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's HLC, so shift by 1
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Previous day's OHLC
    prev_high = high_series.shift(1).values
    prev_low = low_series.shift(1).values
    prev_close = close_series.shift(1).values
    
    # Calculate Camarilla levels for today based on yesterday's price action
    camarilla_p = (prev_high + prev_low + prev_close) / 3
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Volume confirmation: current volume > 2x 20-day average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # need at least 20 days for vol MA + 1 for shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(camarilla_p[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(prev_close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Close breaks above R1 with volume confirmation and above 1w EMA trend
            if close[i] > camarilla_r1[i] and vol_ok and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 with volume confirmation and below 1w EMA trend
            elif close[i] < camarilla_s1[i] and vol_ok and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close crosses back below pivot P
            if close[i] < camarilla_p[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close crosses back above pivot P
            if close[i] > camarilla_p[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals