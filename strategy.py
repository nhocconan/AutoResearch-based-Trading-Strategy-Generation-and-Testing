#!/usr/bin/env python3
"""
12h strategy combining Williams %R with 1d EMA trend filter and volume spike confirmation.
Williams %R identifies overbought/oversold conditions on 12h timeframe.
Entry occurs when %R exits extreme territory (oversold for long, overbought for short)
with confirmation from 1d EMA trend and volume spike.
Exit when %R returns to neutral range or trend changes.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R on 12h (14-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(13, n):
        highest_high[i] = np.max(high_12h[i-13:i+1])
        lowest_low[i] = np.min(low_12h[i-13:i+1])
    
    # Williams %R = -(Highest High - Close) / (Highest High - Lowest Low) * 100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        -(highest_high - close_12h) / (highest_high - lowest_low) * 100,
        -50  # neutral when range is zero
    )
    
    # Volume confirmation: 12h volume / 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r[i]
        ema_1d_val = ema_34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        price_close = close_12h[i]
        
        # Volume spike filter
        vol_threshold = 1.5
        
        if position == 0:
            # Enter long: Williams %R rises above -80 (exiting oversold) + price > 1d EMA + volume spike
            if (williams_r_val > -80 and 
                price_close > ema_1d_val and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R falls below -20 (exiting overbought) + price < 1d EMA + volume spike
            elif (williams_r_val < -20 and 
                  price_close < ema_1d_val and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R falls below -50 (returns from overbought) OR trend fails
                if williams_r_val < -50 or price_close < ema_1d_val:
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R rises above -50 (returns from oversold) OR trend fails
                if williams_r_val > -50 or price_close > ema_1d_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34_Volume_Spike"
timeframe = "12h"
leverage = 1.0