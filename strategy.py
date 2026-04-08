#!/usr/bin/env python3
# 6h_volume_price_action_v1
# Hypothesis: 6h volume-price action strategy with 1d trend filter.
# Long when: price closes above 6h VWAP + volume > 1.5x 20-period average + price > 1d EMA200 (uptrend).
# Short when: price closes below 6h VWAP + volume > 1.5x 20-period average + price < 1d EMA200 (downtrend).
# Exit when price crosses back to 6h VWAP.
# Designed to capture institutional volume-driven moves with trend alignment in both bull and bear markets.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_price_action_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 6h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_tpv, cum_vol, out=np.full_like(cum_tpv, np.nan), where=cum_vol!=0)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(vwap[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below VWAP
            if close[i] < vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above VWAP
            if close[i] > vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # VWAP cross entries with trend filter
            if (close[i] > vwap[i]) and volume_ok and (close[i] > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            elif (close[i] < vwap[i]) and volume_ok and (close[i] < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals