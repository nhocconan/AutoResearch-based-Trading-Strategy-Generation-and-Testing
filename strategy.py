#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R1S1_Breakout_1dTrend_Volume_Spike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Previous day's OHLC for Camarilla calculation (use daily OHLC from 6h data)
    # For 6h bars, we need the previous day's OHLC based on the day boundary
    prev_high = np.roll(high, 4)  # 4 * 6h = 24h (previous day)
    prev_low = np.roll(low, 4)
    prev_close = np.roll(close, 4)
    prev_open = np.roll(prices['open'].values, 4)
    
    # Handle first 4 bars (no previous day)
    prev_high[:4] = high[:4]
    prev_low[:4] = low[:4]
    prev_close[:4] = close[:4]
    prev_open[:4] = prices['open'].values[:4]
    
    # Calculate Camarilla R1 and S1 levels
    range_ = prev_high - prev_low
    close_prev = prev_close
    r1 = close_prev + range_ * 1.1 / 6
    s1 = close_prev - range_ * 1.1 / 6
    
    # Daily trend: EMA34 on 1d (using proper 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume spike > 2.0x 20-period SMA (higher threshold for fewer trades)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure we have enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(r1[i]) or np.isnan(s1[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: breakout above R1 with daily uptrend and volume spike
            if (price > r1[i] and 
                price > ema34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: breakdown below S1 with daily downtrend and volume spike
            elif (price < s1[i] and 
                  price < ema34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to daily EMA or loses volume spike
            if (price < ema34_1d_aligned[i] or 
                not vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to daily EMA or loses volume spike
            if (price > ema34_1d_aligned[i] or 
                not vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals