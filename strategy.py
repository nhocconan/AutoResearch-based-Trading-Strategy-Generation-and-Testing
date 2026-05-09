#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Tight_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(open_, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    prev_open[0] = open_[0]
    
    # Calculate Camarilla R1 and S1 levels
    range_ = prev_high - prev_low
    close_prev = prev_close
    r1 = close_prev + range_ * 1.1 / 6
    s1 = close_prev - range_ * 1.1 / 6
    
    # Daily trend: EMA34 on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.8x 30-period SMA (stricter)
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > 1.8 * vol_ma30
    
    # Range filter: avoid choppy markets
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    range_filter = atr < atr_ma50  # Only trade when volatility is below average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(r1[i]) or np.isnan(s1[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma30[i]) or \
           np.isnan(atr[i]) or np.isnan(atr_ma50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: breakout above R1 with daily uptrend, volume, and low volatility
            if (price > r1[i] and 
                price > ema34_1d_aligned[i] and 
                vol_filter[i] and 
                range_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: breakdown below S1 with daily downtrend, volume, and low volatility
            elif (price < s1[i] and 
                  price < ema34_1d_aligned[i] and 
                  vol_filter[i] and 
                  range_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to daily EMA or loses conditions
            if (price < ema34_1d_aligned[i] or 
                not vol_filter[i] or 
                not range_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to daily EMA or loses conditions
            if (price > ema34_1d_aligned[i] or 
                not vol_filter[i] or 
                not range_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals