#!/usr/bin/env python3
# 12h_OrderBlock_1dTrend_Volume
# Hypothesis: Institutional order blocks on 12h chart provide high-probability reversal zones.
# In trending markets (1d EMA50), price pulling back to bullish/bearish order blocks
# offers continuation trades with favorable risk-reward. Volume confirmation filters
# weak signals. Works in bull markets (pullbacks to bullish OBs in uptrend) and
# bear markets (pullbacks to bearish OBs in downtrend). Target: 15-35 trades/year.

name = "12h_OrderBlock_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h order blocks (bullish/bearish)
    # Bullish OB: strong down candle followed by strong up candle closing above midpoint
    # Bearish OB: strong up candle followed by strong down candle closing below midpoint
    body_size = np.abs(close - open_)
    avg_body = pd.Series(body_size).rolling(window=20, min_periods=20).mean().values
    
    bullish_ob = np.zeros(n, dtype=bool)
    bearish_ob = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        if np.isnan(avg_body[i]) or np.isnan(avg_body[i-1]) or np.isnan(avg_body[i-2]):
            continue
            
        # Current candle
        curr_body = body_size[i]
        curr_mid = (high[i] + low[i]) / 2
        
        # Previous candle
        prev_close = close[i-1]
        prev_open = open_[i-1]
        prev_body = body_size[i-1]
        
        # Two candles ago
        prev2_close = close[i-2]
        prev2_open = open_[i-2]
        prev2_body = body_size[i-2]
        
        # Bullish OB: two red candles followed by strong green closing above midpoint of second red
        if (close[i-2] < open_[i-2] and  # red
            close[i-1] < open_[i-1] and  # red
            close[i] > open_[i] and      # green
            curr_body > avg_body[i] * 1.5 and  # strong body
            close[i] > (high[i-1] + low[i-1]) / 2):  # above midpoint of prev candle
            bullish_ob[i] = True
            
        # Bearish OB: two green candles followed by strong red closing below midpoint of second green
        if (close[i-2] > open_[i-2] and  # green
            close[i-1] > open_[i-1] and  # green
            close[i] < open_[i] and      # red
            curr_body > avg_body[i] * 1.5 and  # strong body
            close[i] < (high[i-1] + low[i-1]) / 2):  # below midpoint of prev candle
            bearish_ob[i] = True
    
    # Volume confirmation (20-period MA on 12h = ~10 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + bullish OB + volume
            if uptrend and bullish_ob[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + bearish OB + volume
            elif downtrend and bearish_ob[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or bearish OB appears
            if not uptrend or bearish_ob[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or bullish OB appears
            if not downtrend or bullish_ob[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals