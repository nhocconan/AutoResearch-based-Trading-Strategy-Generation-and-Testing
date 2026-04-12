# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mlib import talib
from mlib.talib import SMA, STDDEV

# Hypothesis: 4h Bollinger Band breakouts with volume confirmation and trend filter
# Works in bull/bear by using Bollinger Bands (adaptive volatility bands) and requiring
# volume confirmation to avoid false breakouts. Trend filter uses 4h 50 EMA to align with
# intermediate trend. Designed for low trade frequency (<50/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    basis = SMA(close, 20)
    dev = STDDEV(close, 20)
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # 4h 50 EMA for trend filter
    ema50 = talib.EMA(close, 50)
    
    # Volume confirmation: volume > 1.5x 20-period average volume
    avg_vol = SMA(volume, 20)
    vol_threshold = avg_vol * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after indicators are valid
        # Skip if data not ready
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema50[i]
        downtrend = close[i] < ema50[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_threshold[i]
        
        # Breakout conditions
        long_breakout = close[i] > upper[i]
        short_breakout = close[i] < lower[i]
        
        # Entry conditions
        long_entry = long_breakout and uptrend and vol_confirm
        short_entry = short_breakout and downtrend and vol_confirm
        
        # Exit conditions: return to middle band or opposite breakout
        long_exit = close[i] < basis[i] or (position == 1 and close[i] < lower[i])
        short_exit = close[i] > basis[i] or (position == -1 and close[i] > upper[i])
        
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
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_bb_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0