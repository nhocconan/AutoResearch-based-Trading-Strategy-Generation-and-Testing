#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    weekly_ema = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Camarilla pivot levels for 1d timeframe
    high = prices['high'].values
    low = prices['low'].values
    close_prev = np.roll(close, 1)
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    
    # Set first values to avoid NaN
    close_prev[0] = close[0]
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    
    # Calculate Camarilla levels
    range_prev = high_prev - low_prev
    camarilla_h4 = close_prev + 1.1 * range_prev / 2
    camarilla_l4 = close_prev - 1.1 * range_prev / 2
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below weekly EMA
        above_weekly_ema = close[i] > weekly_ema_aligned[i]
        below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price below Camarilla L4 or loss of weekly uptrend
            if close[i] < camarilla_l4[i] or not above_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price above Camarilla H4 or loss of weekly downtrend
            if close[i] > camarilla_h4[i] or not below_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above Camarilla H4 + volume + weekly uptrend
            if (close[i] > camarilla_h4[i] and 
                vol_confirm and 
                above_weekly_ema):
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below Camarilla L4 + volume + weekly downtrend
            elif (close[i] < camarilla_l4[i] and 
                  vol_confirm and 
                  below_weekly_ema):
                position = -1
                signals[i] = -0.25
    
    return signals