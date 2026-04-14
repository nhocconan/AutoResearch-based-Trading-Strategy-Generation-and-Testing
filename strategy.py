#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full(len(df_1d), np.nan)
    lowest_low = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        highest_high[i] = np.max(high_1d[i-13:i+1])
        lowest_low[i] = np.min(low_1d[i-13:i+1])
    
    williams_r = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = ((highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i])) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(df_1d), np.nan)
    avg_loss = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, len(df_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(len(df_1d), np.nan)
    rsi = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Calculate daily ATR (14-period) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[1:15])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align indicators to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    rsi_6h = align_htf_to_ltf(prices, df_1d, rsi)
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6-hour price momentum (rate of change over 3 periods)
    roc_6h = np.full(n, np.nan)
    if n >= 3:
        for i in range(2, n):
            if close[i-3] != 0:
                roc_6h[i] = ((close[i] - close[i-3]) / close[i-3]) * 100
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_6h[i]) or
            np.isnan(rsi_6h[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(roc_6h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_6h[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Williams %R oversold/overbought conditions
        williams_oversold = williams_r_6h[i] <= -80
        williams_overbought = williams_r_6h[i] >= -20
        
        # RSI confirmation (avoid extreme overbought/oversold)
        rsi_not_extreme = (rsi_6h[i] > 20) and (rsi_6h[i] < 80)
        
        # Momentum confirmation
        momentum_up = roc_6h[i] > 0
        momentum_down = roc_6h[i] < 0
        
        if position == 0:
            # Long: Williams %R oversold + RSI not extreme + positive momentum
            if williams_oversold and rsi_not_extreme and momentum_up:
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought + RSI not extreme + negative momentum
            elif williams_overbought and rsi_not_extreme and momentum_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Williams %R returns to neutral OR RSI becomes overbought
            if williams_r_6h[i] >= -50 or rsi_6h[i] >= 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Williams %R returns to neutral OR RSI becomes oversold
            if williams_r_6h[i] <= -50 or rsi_6h[i] <= 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Williams_RSI_Momentum_v1"
timeframe = "6h"
leverage = 1.0