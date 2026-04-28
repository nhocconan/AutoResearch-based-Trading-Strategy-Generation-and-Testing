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
    open_time = prices['open_time'].values
    
    # Pre-compute hour filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume ratio (current 4h volume / 20-period average)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # ATR(14) for volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 4h EMA
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume filter: current 4h volume above average
        volume_filter = volume_4h[i // 4] > vol_ma_20_aligned[i] if i // 4 < len(volume_4h) else False
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_4h_aligned[i] > 0.001 * close[i]  # At least 0.1% ATR
        
        # Entry conditions: breakout with volume and trend
        long_entry = uptrend and close[i] > close[i-1] and volume_filter and vol_filter
        short_entry = downtrend and close[i] < close[i-1] and volume_filter and vol_filter
        
        # Exit conditions: opposite signal or volatility expansion
        long_exit = not uptrend or close[i] < close[i-1] or atr_4h_aligned[i] > 2 * atr_4h_aligned[i-1]
        short_exit = not downtrend or close[i] > close[i-1] or atr_4h_aligned[i] > 2 * atr_4h_aligned[i-1]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_EMA50_Volume_Breakout_Session"
timeframe = "1h"
leverage = 1.0