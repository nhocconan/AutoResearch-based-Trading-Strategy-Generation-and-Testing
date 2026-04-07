# 4h_rsi_volume_breakout_v1
# Hypothesis: 4-hour RSI breakout with volume confirmation and daily trend filter.
# Uses RSI(14) crossing above 60 for long, below 40 for short, with volume > 2x average.
# Daily EMA(50) filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Designed to work in both bull and bear markets by following the daily trend direction.
# Target: 20-40 trades/year to minimize fee drag.

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter (using 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    daily_ema = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(daily_ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x average volume
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: price above/below daily EMA
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI < 40 or loss of daily uptrend
            if rsi[i] < 40 or not above_daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: RSI > 60 or loss of daily downtrend
            if rsi[i] > 60 or not below_daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI > 60 + volume + above daily EMA
            if (rsi[i] > 60 and 
                vol_confirm and 
                above_daily_ema):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI < 40 + volume + below daily EMA
            elif (rsi[i] < 40 and 
                  vol_confirm and 
                  below_daily_ema):
                position = -1
                signals[i] = -0.25
    
    return signals