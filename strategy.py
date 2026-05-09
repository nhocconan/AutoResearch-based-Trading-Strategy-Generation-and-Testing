#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h price range (close to close over 12h period)
    # For 12h timeframe, we use the current bar's high/low as the period range
    # Camarilla levels based on previous 12h bar's range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # first bar
    prev_high = np.roll(high, 1)
    prev_high[0] = high[0]
    prev_low = np.roll(low, 1)
    prev_low[0] = low[0]
    
    # Calculate Camarilla levels for R1 and S1
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    range_val = prev_high - prev_low
    r1 = prev_close + range_val * 1.1 / 12
    s1 = prev_close - range_val * 1.1 / 12
    
    # Daily trend filter: EMA34 on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 + daily uptrend + volume confirmation
            if (price > r1[i] and 
                price > ema34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S1 + daily downtrend + volume confirmation
            elif (price < s1[i] and 
                  price < ema34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns below S1 or daily trend turns down
            if (price < s1[i] or 
                price < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above R1 or daily trend turns up
            if (price > r1[i] or 
                price > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals