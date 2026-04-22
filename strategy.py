#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h momentum breakout with daily trend filter
    # Uses price breaking above/below 12-period high/low on 12h chart
    # Confirmed by daily EMA trend direction and volume surge
    # Works in both bull and bear markets by capturing momentum bursts
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA34 trend filter
    ema_1d_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # 12h price channel (12-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12-period highest high and lowest low
    highest_high = pd.Series(high).rolling(window=12, min_periods=12).max().values
    lowest_low = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    # Volume filter (12-period average surge)
    vol_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_surge = volume > 1.5 * vol_ma12
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(12, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 12-period high with volume surge AND daily EMA34 uptrend
            if close[i] > highest_high[i] and vol_surge[i] and close[i] > ema_1d_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12-period low with volume surge AND daily EMA34 downtrend
            elif close[i] < lowest_low[i] and vol_surge[i] and close[i] < ema_1d_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite side of the channel
            if position == 1:
                if close[i] < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Momentum_Breakout_1dEMA34_Trend_VolumeSurge_v1"
timeframe = "12h"
leverage = 1.0