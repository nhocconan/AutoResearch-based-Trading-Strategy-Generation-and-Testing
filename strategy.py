#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeFilter"
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
    
    # Calculate previous day's OHLC for Camarilla levels (using daily data)
    prev_high = np.roll(high, 96)  # 96 * 15min = 24h (previous day)
    prev_low = np.roll(low, 96)
    prev_close = np.roll(close, 96)
    prev_high[:96] = high[:96]  # fill first day with current values
    prev_low[:96] = low[:96]
    prev_close[:96] = close[:96]
    
    # Calculate Camarilla R1 and S1 levels (narrower bands for fewer trades)
    range_ = prev_high - prev_low
    r1 = prev_close + range_ * 1.1 / 12
    s1 = prev_close - range_ * 1.1 / 12
    
    # 12h trend: EMA50 on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 1.5x 30-period SMA
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > 1.5 * vol_ma30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(96, 50, 30)  # ensure all data available
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(r1[i]) or np.isnan(s1[i]) or \
           np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma30[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: breakout above R1 with 12h uptrend and volume
            if (price > r1[i] and 
                price > ema50_12h_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 with 12h downtrend and volume
            elif (price < s1[i] and 
                  price < ema50_12h_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 12h EMA or loses volume
            if (price < ema50_12h_aligned[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 12h EMA or loses volume
            if (price > ema50_12h_aligned[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals