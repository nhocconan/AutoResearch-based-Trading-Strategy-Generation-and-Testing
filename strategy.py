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
    
    # Get weekly data for EMA10 (weekly)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA10 on weekly data
    close_1w_series = pd.Series(close_1w)
    ema10_1w = close_1w_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align weekly EMA10 to daily timeframe
    ema10_1d = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Calculate daily EMA20 for trend filter
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate RSI(14) on daily
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need EMA20, weekly EMA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema10_1d[i]) or np.isnan(ema20[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price above weekly EMA10, EMA20 trending up, RSI not overbought, with volume
            if (close[i] > ema10_1d[i] and 
                ema20[i] > ema20[i-1] and 
                rsi[i] < 70 and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price below weekly EMA10, EMA20 trending down, RSI not oversold, with volume
            elif (close[i] < ema10_1d[i] and 
                  ema20[i] < ema20[i-1] and 
                  rsi[i] > 30 and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below EMA20 or RSI overbought
            if close[i] < ema20[i] or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA20 or RSI oversold
            if close[i] > ema20[i] or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA10_EMA20_RSI_Volume_Filter"
timeframe = "1d"
leverage = 1.0