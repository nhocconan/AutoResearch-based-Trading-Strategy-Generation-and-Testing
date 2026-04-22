#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h trend following with weekly EMA filter and volume confirmation
# Uses weekly EMA20 as trend filter, 1d EMA34 for entry confirmation, and volume spike
# Designed for low trade frequency (~20-30/year) to avoid fee drag
# Works in both bull and bear markets by following the weekly trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Load daily data for entry filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA34 for entry confirmation
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike using 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_20w = ema_20_1w_aligned[i]
        ema_34d = ema_34_1d_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price above weekly EMA20 AND daily EMA34 AND volume spike
            if price > ema_20w and price > ema_34d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price below weekly EMA20 AND daily EMA34 AND volume spike
            elif price < ema_20w and price < ema_34d and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: trend reversal or loss of momentum
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price falls below weekly EMA20 or daily EMA34
                if price < ema_20w or price < ema_34d:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price rises above weekly EMA20 or daily EMA34
                if price > ema_20w or price > ema_34d:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WeeklyEMA20_DailyEMA34_Volume"
timeframe = "12h"
leverage = 1.0