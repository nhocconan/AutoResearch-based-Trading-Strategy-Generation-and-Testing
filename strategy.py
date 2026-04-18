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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-day EMA on daily
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    else:
        ema_50_1d = np.full_like(close_1d, np.nan)
    
    # Align 1d EMA to 1h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period volume average on 4h
    vol_ma_4h = np.full_like(volume_4h, np.nan)
    vol_period = 20
    
    if len(volume_4h) >= vol_period:
        for i in range(vol_period, len(volume_4h)):
            vol_ma_4h[i] = np.mean(volume_4h[i-vol_period:i])
    
    # Align 4h volume average to 1h
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 1
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average on 4h
        vol_confirm = volume[i] > 1.5 * vol_ma_4h_aligned[i]
        
        # Trend filter: price above/below 50-day EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price above 50-day EMA with volume confirmation
            if uptrend and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price below 50-day EMA with volume confirmation
            elif downtrend and vol_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 50-day EMA
            if not uptrend:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 50-day EMA
            if not downtrend:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_50dEMA_Volume_Session"
timeframe = "1h"
leverage = 1.0