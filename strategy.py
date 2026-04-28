#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily RSI(14) for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    alpha = 1/14
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=alpha, adjust=False, min_periods=14).mean().values
    avg_loss = loss_series.ewm(alpha=alpha, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.where(avg_loss == 0, 100, rsi_1d)
    
    # Calculate daily volume average for volume filter
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 12h Donchian channels (10-period - tighter for fewer trades)
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 12h EMA(25) for trend filter
    ema_25 = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(high_10[i]) or 
            np.isnan(low_10[i]) or 
            np.isnan(ema_25[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA25 for long, below for short
        uptrend = close[i] > ema_25[i]
        downtrend = close[i] < ema_25[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_10[i-1]  # Break above previous high
        breakout_down = close[i] < low_10[i-1]  # Break below previous low
        
        # RSI momentum filter: avoid overbought/oversold extremes
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        # Volume filter: ensure sufficient volume
        vol_ok = volume[i] > vol_avg_aligned[i] * 0.5
        
        # Long conditions: uptrend + breakout up + RSI not overbought + volume
        long_condition = uptrend and breakout_up and rsi_not_overbought and vol_ok
        
        # Short conditions: downtrend + breakout down + RSI not oversold + volume
        short_condition = downtrend and breakout_down and rsi_not_oversold and vol_ok
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout
        elif position == 1 and close[i] < low_10[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > high_10[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian10_Breakout_EMA25_RSI_Volume"
timeframe = "12h"
leverage = 1.0