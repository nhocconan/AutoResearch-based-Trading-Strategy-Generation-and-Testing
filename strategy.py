#!/usr/bin/env python3
# 4h_RSI_MeanReversion_With_Volume_And_Trend
# Hypothesis: RSI mean reversion with volume confirmation and trend filter on 4h timeframe.
# Works in bull/bear: Trend filter (EMA50) avoids counter-trend trades, volume confirms reversals.
# RSI(14) < 30 for long, > 70 for short. Uses 1d EMA50 for trend filter.
# Target: 20-50 trades/year with low turnover to minimize fee drag.

name = "4h_RSI_MeanReversion_With_Volume_And_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    # Wilder's smoothing (equivalent to alpha=1/14)
    alpha = 1.0 / 14
    for i in range(len(gain)):
        if i == 0:
            avg_gain[i] = gain[i]
            avg_loss[i] = loss[i]
        else:
            avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
            avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # Handle no loss case
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_filter = np.full_like(volume, np.nan)
    valid_ma = ~np.isnan(vol_ma) & (vol_ma > 0)
    volume_filter[valid_ma] = volume[valid_ma] > (1.5 * vol_ma[valid_ma])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 50)  # RSI(14), vol MA(20), EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI < 30 (oversold) AND uptrend (price > EMA50) AND volume confirmation
            if (rsi[i] < 30 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 (overbought) AND downtrend (price < EMA50) AND volume confirmation
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) OR trend reversal
            if rsi[i] > 50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) OR trend reversal
            if rsi[i] < 50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals