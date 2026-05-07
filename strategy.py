#!/usr/bin/env python3
"""
4h_RSI_Divergence_1dTrend_Filter
Hypothesis: Trade RSI divergences on 4h timeframe filtered by 1d trend (EMA200) and volume confirmation. RSI divergences capture reversals in both bull and bear markets, while 1d trend filter ensures alignment with higher timeframe momentum. Volume spike confirms conviction. Designed for 20-40 trades/year to minimize fee drag.
"""

name = "4h_RSI_Divergence_1dTrend_Filter"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d EMA200 for trend filter
    daily_close = df_1d['close'].values
    ema_200_1d = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Align daily close for trend comparison
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Get 4h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup for RSI and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(daily_close_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = daily_close_aligned[i] > ema_200_1d_aligned[i]
        trend_down = daily_close_aligned[i] < ema_200_1d_aligned[i]
        
        # Check for RSI divergence (look back 5 bars for swing high/low)
        lookback = 5
        if i >= lookback:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (low[i] < low[i-lookback] and 
                rsi[i] > rsi[i-lookback] and
                trend_up and 
                vol_ratio[i] > 1.5):
                if position == 0:
                    signals[i] = 0.30
                    position = 1
                elif position == -1:
                    signals[i] = 0.30  # reverse to long
                    position = 1
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (high[i] > high[i-lookback] and 
                  rsi[i] < rsi[i-lookback] and
                  trend_down and 
                  vol_ratio[i] > 1.5):
                if position == 0:
                    signals[i] = -0.30
                    position = -1
                elif position == 1:
                    signals[i] = -0.30  # reverse to short
                    position = -1
        
        # Exit conditions
        if position == 1:
            # Exit long: RSI overbought or trend turns down
            if rsi[i] > 70 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: RSI oversold or trend turns up
            if rsi[i] < 30 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals