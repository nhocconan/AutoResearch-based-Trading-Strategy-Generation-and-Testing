#!/usr/bin/env python3
# 12h_1d_ema_rsi_volume_v1
# Hypothesis: Use 1d EMA20 for trend direction, 1d RSI(14) for momentum, and volume surge on 12h for entry.
# Long when: 12h close > 1d EMA20, 1d RSI > 50, and volume > 2x 20-period average.
# Short when: 12h close < 1d EMA20, 1d RSI < 50, and volume > 2x 20-period average.
# Exit when: 12h close crosses 1d EMA20 in opposite direction.
# Uses 1d trend/momentum filter with 12h execution to avoid overtrading.
# Target: 15-30 trades/year with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ema_rsi_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA20 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_20_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_20)
    
    # 1d RSI(14) momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    period = 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume filter: 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1d_20_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below 1d EMA20
            if close[i] < ema_1d_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above 1d EMA20
            if close[i] > ema_1d_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above 1d EMA20, 1d RSI > 50, volume surge
            if (close[i] > ema_1d_20_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below 1d EMA20, 1d RSI < 50, volume surge
            elif (close[i] < ema_1d_20_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals