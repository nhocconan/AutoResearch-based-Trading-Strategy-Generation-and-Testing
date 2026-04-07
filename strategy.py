#!/usr/bin/env python3
"""
6h_ema_slope_volume_breakout
Hypothesis: On 6h timeframe, use EMA slope (rate of change) to detect strong momentum trends combined with volume confirmation. Enter long when EMA20 slope turns positive with price above EMA50 and volume > 2x average; enter short when EMA20 slope turns negative with price below EMA50 and volume > 2x average. Exit when EMA slope reverses or price crosses opposite EMA. This captures institutional momentum moves while avoiding chop. Works in bull/bear via slope direction filter. Targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema_slope_volume_breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMAs
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate EMA20 slope (rate of change over 3 periods)
    ema20_slope = (ema20 - np.roll(ema20, 3)) / 3
    ema20_slope[:3] = 0  # First 3 values invalid
    
    # Volume confirmation (24-period average on 6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(ema20_slope[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 24-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if EMA20 slope turns negative (momentum loss)
            if ema20_slope[i] < 0:
                exit_long = True
            # Exit if price crosses below EMA50 (trend change)
            elif close[i] < ema50[i] and close[i-1] >= ema50[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if EMA20 slope turns positive (momentum loss)
            if ema20_slope[i] > 0:
                exit_short = True
            # Exit if price crosses above EMA50 (trend change)
            elif close[i] > ema50[i] and close[i-1] <= ema50[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA20 slope turns positive, price above EMA50, volume confirmation
            long_entry = False
            if (ema20_slope[i] > 0 and ema20_slope[i-1] <= 0 and
                close[i] > ema50[i] and vol_confirm):
                long_entry = True
            
            # Short entry: EMA20 slope turns negative, price below EMA50, volume confirmation
            short_entry = False
            if (ema20_slope[i] < 0 and ema20_slope[i-1] >= 0 and
                close[i] < ema50[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals