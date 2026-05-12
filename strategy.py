#!/usr/bin/env python3
name = "4h_VWAP_Breakout_Trend_200_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate VWAP on 4h
    vwap = (high + low + close) / 3
    # Typical price * volume
    tpv = vwap * volume
    # Cumulative sums
    cum_tpv = np.cumsum(tpv)
    cum_vol = np.cumsum(volume)
    # VWAP calculation with division by zero protection
    vwap_values = np.divide(cum_tpv, cum_vol, out=np.full_like(cum_tpv, np.nan), where=cum_vol!=0)
    
    # 1d trend filter: EMA(200) on daily close
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # ensure EMA200 has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(vwap_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP + daily uptrend + volume spike
            if (close[i] > vwap_values[i] and 
                close[i] > ema200_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP + daily downtrend + volume spike
            elif (close[i] < vwap_values[i] and 
                  close[i] < ema200_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below VWAP
            if close[i] < vwap_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above VWAP
            if close[i] > vwap_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals