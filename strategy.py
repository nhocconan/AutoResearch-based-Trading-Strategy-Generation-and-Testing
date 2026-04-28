# 1h_RSI_Volume_Trend_Filter
# Hypothesis: On 1h timeframe, combine RSI mean reversion with volume confirmation and 4h EMA trend filter.
# In bull markets: RSI oversold during uptrend signals continuation longs. In bear markets: RSI overbought during downtrend signals continuation shorts.
# Volume filter ensures breakouts have institutional participation. 4h EMA filter prevents counter-trend trades.
# Target: 20-50 trades per year to minimize fee drag on 1h timeframe.
# Uses RSI(14) on 1h, volume confirmation, and 4h EMA(50) trend filter for multi-timeframe alignment.

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(50) for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter from 4h EMA50
        uptrend = close[i] > ema50_4h_aligned[i]
        downtrend = close[i] < ema50_4h_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # RSI conditions: oversold in uptrend, overbought in downtrend
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        long_entry = rsi_oversold and uptrend and vol_filter
        short_entry = rsi_overbought and downtrend and vol_filter
        
        # Exit conditions: RSI returns to neutral or trend reverses
        long_exit = (rsi[i] > 50) or (not uptrend)
        short_exit = (rsi[i] < 50) or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_RSI_Volume_Trend_Filter"
timeframe = "1h"
leverage = 1.0