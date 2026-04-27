#!/usr/bin/env python3
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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    
    # Calculate daily SMA(50) for trend filter
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate daily ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 15-minute timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate RSI moving average for smoothing
    rsi_ma = pd.Series(rsi).rolling(window=3, min_periods=3).mean().values
    rsi_ma_aligned = align_htf_to_ltf(prices, df_1d, rsi_ma)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(sma_50_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(rsi_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above SMA50 for long, below for short
        uptrend = close[i] > sma_50_aligned[i]
        downtrend = close[i] < sma_50_aligned[i]
        
        # RSI conditions: oversold/overbought with confirmation
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        rsi_rising = rsi_ma_aligned[i] > rsi_aligned[i-1] if i > 0 else False
        rsi_falling = rsi_ma_aligned[i] < rsi_aligned[i-1] if i > 0 else False
        
        # Volatility filter: avoid extremely low volatility
        atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
        atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
        if np.isnan(atr_ma_aligned[i]):
            signals[i] = 0.0
            continue
        vol_ok = atr_aligned[i] > (atr_ma_aligned[i] * 0.3)
        
        # Long conditions: uptrend + RSI oversold + rising RSI + volatility ok
        long_condition = uptrend and rsi_oversold and rsi_rising and vol_ok
        
        # Short conditions: downtrend + RSI overbought + falling RSI + volatility ok
        short_condition = downtrend and rsi_overbought and rsi_falling and vol_ok
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite RSI extreme
        elif position == 1 and rsi_aligned[i] > 50:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi_aligned[i] < 50:
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

name = "15m_RSI_SMA50_Trend_Filter"
timeframe = "15m"
leverage = 1.0