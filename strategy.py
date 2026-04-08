#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_ema_bounce_volume_v1"
timeframe = "1d"
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
    
    # 1w data for trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA 34 (trend filter)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily EMA 21 (dynamic support/resistance)
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_21[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA21 or weekly trend turns bearish
            if close[i] < ema_21[i] or close[i] < ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA21 or weekly trend turns bullish
            if close[i] > ema_21[i] or close[i] > ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Weekly trend filter
            weekly_bullish = close[i] > ema_34_1w_aligned[i]
            weekly_bearish = close[i] < ema_34_1w_aligned[i]
            
            # Long: price bounces off EMA21 in bullish weekly trend + volume
            if (close[i] > ema_21[i] and 
                low[i] <= ema_21[i] * 1.002 and  # touched or slightly below EMA21
                weekly_bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price rejected from EMA21 in bearish weekly trend + volume
            elif (close[i] < ema_21[i] and 
                  high[i] >= ema_21[i] * 0.998 and  # touched or slightly above EMA21
                  weekly_bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals