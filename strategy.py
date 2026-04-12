# 4h_1d_ema_slope_breakout_v1
# 4-hour EMA slope with 1-day EMA direction filter and volume confirmation
# Captures breakouts in trending markets while avoiding false signals in chop
# EMA slope indicates momentum strength; 1-day EMA direction provides trend filter
# Volume confirms institutional participation
# Target: 20-40 trades/year per symbol for low friction and good generalization

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_slope_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA slope: rate of change of 20-period EMA
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_slope = np.diff(ema_20, prepend=ema_20[0])  # daily change in EMA value
    
    # 1-day EMA direction: 50-period EMA slope (trend filter)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_slope = np.diff(ema_50_1d, prepend=ema_50_1d[0])
    ema_50_1d_dir = align_htf_to_ltf(prices, df_1d, ema_50_1d_slope)  # positive = uptrend
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema_slope[i]) or np.isnan(ema_50_1d_dir[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # EMA slope threshold: significant momentum
        slope_threshold = ema_20[i] * 0.001  # 0.1% of EMA value as threshold
        
        # Long conditions: positive slope + uptrend + volume
        if (ema_slope[i] > slope_threshold and 
            ema_50_1d_dir[i] > 0 and 
            vol_confirm[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short conditions: negative slope + downtrend + volume
        elif (ema_slope[i] < -slope_threshold and 
              ema_50_1d_dir[i] < 0 and 
              vol_confirm[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: slope reverses or volume dries up
        elif ((ema_slope[i] < 0 and position == 1) or 
              (ema_slope[i] > 0 and position == -1) or
              not vol_confirm[i]):
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