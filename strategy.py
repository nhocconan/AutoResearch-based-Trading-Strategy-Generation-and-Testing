#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_trend_follow_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(open_time).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_50_4h = ema(close_4h, 50)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(200) on 1d
    ema_200_1d = ema(close_1d, 200)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(200, 50)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        if not session_mask[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below 4h EMA50 or stoploss hit
            if (close[i] < ema_50_4h_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price closes above 4h EMA50 or stoploss hit
            if (close[i] > ema_50_4h_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries - only long in bull, only short in bear based on 1d trend
            # Long: price above 4h EMA50, above 1d EMA200, with volume
            if (close[i] > ema_50_4h_aligned[i] and 
                close[i] > ema_200_1d_aligned[i] and 
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price below 4h EMA50, below 1d EMA200, with volume
            elif (close[i] < ema_50_4h_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals