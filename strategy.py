#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_Energy_12hTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 on 12h close for trend
    close_12h = df_12h['close'].values
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_13_12h)
    
    # Get 1d data for Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = df_1d['high'].values - ema_13_1d
    # Bear Power = Low - EMA13
    bear_power = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume filter: current volume > 1.3 * 20-period average (less strict to allow more trades)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_13_12h_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_12h = ema_13_12h_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 (bullish energy) AND price above 12h EMA13 AND volume filter
            if bull_power_val > 0 and close[i] > ema_12h and vol_filt:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 (bearish energy) AND price below 12h EMA13 AND volume filter
            elif bear_power_val < 0 and close[i] < ema_12h and vol_filt:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power < 0 (bearish energy takes over) OR trend breaks (price < 12h EMA13)
            if bear_power_val < 0 or close[i] < ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power > 0 (bullish energy takes over) OR trend breaks (price > 12h EMA13)
            if bull_power_val > 0 or close[i] > ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals