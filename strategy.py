#!/usr/bin/env python3
"""
4h_RSI_Extreme_TrendFilter_V1
Strategy: 4h RSI(14) extremes with daily EMA50 trend filter and volume confirmation.
Long: RSI < 30 + price > daily EMA50 + volume > 1.5x 20-period average
Short: RSI > 70 + price < daily EMA50 + volume > 1.5x 20-period average
Exit: Opposite RSI extreme or trend reversal
Position size: 0.25
Designed to capture mean-reversion in ranging markets while respecting trend.
Timeframe: 4h
"""

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
    def rsi(close, window=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=window, min_periods=window).mean().values
        avg_loss = pd.Series(loss).rolling(window=window, min_periods=window).mean().values
        
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0), where=avg_loss!=0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    rsi_val = rsi(close, 14)
    
    # Calculate daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume average (20-period)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(rsi_val[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.5 * volume_ma20_4h_aligned[i])
        
        # Trend filter: price above/below daily EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Entry signals
        if position == 0:
            # Long: RSI < 30 (oversold) + volume filter + trend up
            if rsi_val[i] < 30 and volume_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + volume filter + trend down
            elif rsi_val[i] > 70 and volume_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 70 (overbought) or trend down
            if rsi_val[i] > 70 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 30 (oversold) or trend up
            if rsi_val[i] < 30 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Extreme_TrendFilter_V1"
timeframe = "4h"
leverage = 1.0