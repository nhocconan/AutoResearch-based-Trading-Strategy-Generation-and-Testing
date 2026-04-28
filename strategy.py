#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly SMA(20) for trend filter
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Get daily data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily volume SMA(20)
    vol_sma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)
    
    # Calculate daily RSI(14)
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
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Daily Donchian channels (20-period) for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_20_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_sma_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above weekly SMA20 for long, below for short
        uptrend = close[i] > sma_20_1w_aligned[i]
        downtrend = close[i] < sma_20_1w_aligned[i]
        
        # Daily Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous high
        breakout_down = close[i] < low_20[i-1]  # Break below previous low
        
        # Volume filter: volume above average
        vol_ok = volume[i] > vol_sma_1d_aligned[i]
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_ok_long = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        rsi_ok_short = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        # Volatility filter: ensure sufficient ATR
        vol_filter = atr_1d_aligned[i] > 0.5  # Minimum ATR threshold
        
        # Long conditions: weekly uptrend + daily breakout up + volume + RSI OK + volatility
        long_condition = uptrend and breakout_up and vol_ok and rsi_ok_long and vol_filter
        
        # Short conditions: weekly downtrend + daily breakout down + volume + RSI OK + volatility
        short_condition = downtrend and breakout_down and vol_ok and rsi_ok_short and vol_filter
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout
        elif position == 1 and close[i] < low_20[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > high_20[i-1]:
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

name = "1d_WeeklySMA20_Trend_Donchian20_Breakout_VolumeRSI"
timeframe = "1d"
leverage = 1.0