#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Camarilla pivot levels (resistance/support)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r1 = close_1d + (range_ * 1.1 / 12)
    s1 = close_1d - (range_ * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if Camarilla or EMA data not ready
        if np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(ema34_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 + above 1d EMA34 + volume confirmation
            if (close[i] > r1_12h[i]) and (close[i] > ema34_12h[i]) and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + below 1d EMA34 + volume confirmation
            elif (close[i] < s1_12h[i]) and (close[i] < ema34_12h[i]) and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below S1 or below 1d EMA34
            if (close[i] < s1_12h[i]) or (close[i] < ema34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above R1 or above 1d EMA34
            if (close[i] > r1_12h[i]) or (close[i] > ema34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals