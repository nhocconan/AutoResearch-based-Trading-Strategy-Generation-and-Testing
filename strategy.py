#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour Moving Average Trend with Volume Confirmation
# Uses 4h EMA20 as trend filter (above = long bias, below = short bias)
# Enters on 1h pullbacks to EMA20 with volume spike confirmation
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend)
# Target: 60-150 total trades over 4 years = 15-37/year for 1h

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    
    # Load 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h EMA20 to 1h timeframe (wait for 4h bar close)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h EMA20 for pullback entries
    ema_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike detector (volume > 1.5x 20-period median)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(20, n):
        # Skip if required data is NaN or outside trading session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1h[i]) or 
            hours[i] < 8 or hours[i] > 20):
            continue
            
        # Long entry: 4h uptrend + price pulls back to 1h EMA20 + volume spike
        if (ema_4h_aligned[i] > ema_4h_aligned[i-1] and  # 4h EMA rising
            close[i] >= ema_1h[i] and                     # Price at or above 1h EMA
            close[i-1] < ema_1h[i-1] and                  # Was below 1h EMA (pullback)
            volume[i] > 1.5 * vol_median[i] and           # Volume spike
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: 4h downtrend + price rallies to 1h EMA20 + volume spike
        elif (ema_4h_aligned[i] < ema_4h_aligned[i-1] and  # 4h EMA falling
              close[i] <= ema_1h[i] and                     # Price at or below 1h EMA
              close[i-1] > ema_1h[i-1] and                  # Was above 1h EMA (pullback rally)
              volume[i] > 1.5 * vol_median[i] and           # Volume spike
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: trend reversal on 4h EMA
        elif position == 1 and ema_4h_aligned[i] < ema_4h_aligned[i-1]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and ema_4h_aligned[i] > ema_4h_aligned[i-1]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_4h_EMA20_Pullback_Volume"
timeframe = "1h"
leverage = 1.0