#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1_S1_Breakout_Trend_Volume"
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
    
    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R1, S1)
    r1 = prev_close + 1.1 * (prev_high - prev_low) / 4
    s1 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Weekly trend filter: 20-period SMA
    sma20_1w = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: current daily volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.5)
    
    # Align all to daily
    r1_daily = align_htf_to_ltf(prices, df_1d, r1)
    s1_daily = align_htf_to_ltf(prices, df_1d, s1)
    sma20_1w_daily = align_htf_to_ltf(prices, df_1w, sma20_1w)
    volume_filter_daily = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for SMA20 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or
            np.isnan(sma20_1w_daily[i]) or np.isnan(volume_filter_daily[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_daily[i]
        s1_val = s1_daily[i]
        trend = sma20_1w_daily[i]
        vol_filter = volume_filter_daily[i]
        
        if position == 0:
            # Enter long: break above R1 with volume and above weekly trend
            if close[i] > r1_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume and below weekly trend
            elif close[i] < s1_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 (mean reversion to center)
            if close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R1 (mean reversion to center)
            if close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals