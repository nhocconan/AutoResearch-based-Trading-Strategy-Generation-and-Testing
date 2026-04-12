#!/usr/bin/env python3
"""
6h_1d_Angle_Momentum_With_Volume_v1
Hypothesis: Combine price angle (6-period ROC) with 1d trend filter and volume confirmation.
Go long when price angle > 0.5%, 1d close > 1d EMA50, and volume > 1.5x 20-period average.
Go short when price angle < -0.5%, 1d close < 1d EMA50, and volume > 1.5x average.
Exit when angle reverses or volume drops. Designed for 6h timeframe to capture medium-term
momentum with institutional validation from daily trend and volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
Works in bull via upward angle + uptrend, in bear via downward angle + downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Angle_Momentum_With_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Price angle: 6-period ROC as percentage
    price_change = close - np.roll(close, 6)
    price_change[:6] = 0  # first 6 bars have no 6-period lookback
    price_close_6 = np.roll(close, 6)
    price_close_6[:6] = close[:6]  # avoid division by zero
    price_angle = np.divide(price_change, price_close_6, 
                           out=np.zeros_like(price_change), 
                           where=price_close_6!=0) * 100
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(price_angle[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        long_entry = (price_angle[i] > 0.5 and 
                     close[i] > ema50_1d_aligned[i] and
                     vol_ratio[i] > 1.5)
        short_entry = (price_angle[i] < -0.5 and 
                      close[i] < ema50_1d_aligned[i] and
                      vol_ratio[i] > 1.5)
        
        # Exit conditions: angle reverses or volume drops
        long_exit = price_angle[i] < 0 or vol_ratio[i] < 1.2
        short_exit = price_angle[i] > 0 or vol_ratio[i] < 1.2
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals