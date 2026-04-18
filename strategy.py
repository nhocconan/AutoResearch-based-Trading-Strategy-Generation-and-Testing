#!/usr/bin/env python3
"""
6h Elder Ray Power + Weekly Trend Filter
Uses Elder Ray Bull/Bear Power from 1d to gauge institutional buying/selling pressure,
combined with 1-week EMA trend filter. Takes long when bull power > 0 and above weekly EMA,
short when bear power < 0 and below weekly EMA. Includes volume confirmation to avoid false signals.
Designed for low trade frequency with edge in both bull (bull power persistence) and bear (bear power persistence) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation (EMA13)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for 1d (used in Elder Ray)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray powers to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1-week EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike detection (1.5x 6-period average to reduce noise)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        ema_trend = ema34_1w_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: bull power positive (buying pressure) + price above weekly EMA + volume spike
            if (bull_power > 0 and 
                price > ema_trend and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bear power negative (selling pressure) + price below weekly EMA + volume spike
            elif (bear_power < 0 and 
                  price < ema_trend and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: hold while bull power remains positive
            signals[i] = 0.25
            # Exit: bull power turns negative (selling pressure taking over)
            if bull_power <= 0:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold while bear power remains negative
            signals[i] = -0.25
            # Exit: bear power turns positive (buying pressure taking over)
            if bear_power >= 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Power_WeeklyEMA34_Volume"
timeframe = "6h"
leverage = 1.0