#!/usr/bin/env python3
"""
6h_ema200_rsi14_volume_v1
Hypothesis: On 6h timeframe, use EMA200 trend filter + RSI14 mean reversion + volume confirmation.
Long when price > EMA200 (uptrend) and RSI < 30 (oversold) with volume > 1.5x average.
Short when price < EMA200 (downtrend) and RSI > 70 (overbought) with volume > 1.5x average.
Exit when RSI returns to neutral (40-60 range).
Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
Target: 15-30 trades/year to avoid fee drag while capturing mean reversion within trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema200_rsi14_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA200 for trend filter (using 1d data for stability)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_ema200 = pd.Series(daily_close).ewm(span=200, min_periods=200, adjust=False).mean().values
    daily_ema200_6h = align_htf_to_ltf(prices, df_daily, daily_ema200)
    
    # Calculate RSI14 on 6h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # Handle no loss case
    rsi = np.where(avg_gain == 0, 0, rsi)    # Handle no gain case
    
    # Volume confirmation: 20-period average on 6h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(daily_ema200_6h[i]) or np.isnan(rsi[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral range (40-60)
            if rsi[i] >= 40 and rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral range (40-60)
            if rsi[i] >= 40 and rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Trend filter
            uptrend = close[i] > daily_ema200_6h[i]
            downtrend = close[i] < daily_ema200_6h[i]
            
            # Long entry: uptrend + RSI oversold + volume
            if uptrend and rsi[i] < 30 and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: downtrend + RSI overbought + volume
            elif downtrend and rsi[i] > 70 and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals