#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ElderRay_Power_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray, trend, and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Trend filter: 1d EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current 1d volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.5)
    
    # Align all to 6h
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(ema34_1d_6h[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull = bull_power_6h[i]
        bear = bear_power_6h[i]
        trend = ema34_1d_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, above trend, volume
            if bull > 0 and bear < 0 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Bull Power < 0, Bear Power > 0, below trend, volume
            elif bull < 0 and bear > 0 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power > 0 (bearish pressure)
            if bear > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power > 0 (bullish pressure)
            if bull > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals