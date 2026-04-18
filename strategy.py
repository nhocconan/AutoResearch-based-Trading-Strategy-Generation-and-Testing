#!/usr/bin/env python3
"""
1h_RSI_MeanReversion_4hTrend
Hypothesis: Uses 4h EMA20 for trend direction and 1h RSI(14) for mean-reversion entries. 
In uptrend (price > 4h EMA20), buy when RSI < 30; in downtrend (price < 4h EMA20), sell when RSI > 70.
Adds volume confirmation (volume > 1.5x 20-period average) and session filter (08-20 UTC) to reduce false signals.
Designed for 1h timeframe with tight entry conditions to limit trades and avoid fee drag.
Works in both bull and bear markets by following higher-timeframe trend while exploiting short-term mean reversion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20
    close_4h = df_4h['close'].values
    ema_20_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 20:
        k = 2 / (20 + 1)
        ema_20_4h[19] = np.mean(close_4h[:20])
        for i in range(20, len(close_4h)):
            ema_20_4h[i] = close_4h[i] * k + ema_20_4h[i-1] * (1 - k)
    
    # Align 4h EMA20 to 1h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, n):
            avg_gain[i] = (gain[i] * 13 + avg_gain[i-1]) / 14
            avg_loss[i] = (loss[i] * 13 + avg_loss[i-1]) / 14
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for RSI and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: uptrend + oversold RSI + volume spike + session
            if (close[i] > ema_20_aligned[i] and 
                rsi[i] < 30 and 
                vol_spike[i] and 
                in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short: downtrend + overbought RSI + volume spike + session
            elif (close[i] < ema_20_aligned[i] and 
                  rsi[i] > 70 and 
                  vol_spike[i] and 
                  in_session[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: RSI > 50 or trend reversal
            if rsi[i] > 50 or close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: RSI < 50 or trend reversal
            if rsi[i] < 50 or close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend"
timeframe = "1h"
leverage = 1.0