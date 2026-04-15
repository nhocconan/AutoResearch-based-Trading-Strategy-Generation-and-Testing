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
    
    # Load weekly data once
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 100:
        return np.zeros(n)
    
    # Weekly close array
    close_weekly = df_weekly['close'].values
    
    # Weekly RSI(14)
    delta = pd.Series(close_weekly).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_weekly = 100 - (100 / (1 + rs))
    rsi_weekly = rsi_weekly.values
    
    # Weekly EMA(50) for trend
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Get aligned weekly indicators
        rsi_aligned = align_htf_to_ltf(prices, df_weekly, rsi_weekly)[i]
        ema50_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)[i]
        
        # Check for NaN values
        if np.isnan(rsi_aligned) or np.isnan(ema50_aligned):
            continue
        
        # Weekly trend filter: price > EMA50 = bullish, price < EMA50 = bearish
        if close[i] > ema50_aligned:
            # Bullish bias: look for RSI oversold bounce
            if rsi_aligned < 30 and position <= 0:
                position = 1
                signals[i] = position_size
            elif rsi_aligned > 70 and position >= 0:
                position = -1
                signals[i] = -position_size
        else:
            # Bearish bias: look for RSI overbought rejection
            if rsi_aligned > 70 and position >= 0:
                position = -1
                signals[i] = -position_size
            elif rsi_aligned < 30 and position <= 0:
                position = 1
                signals[i] = position_size
        
        # Exit conditions
        if position == 1 and rsi_aligned > 50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi_aligned < 50:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WeeklyRSI_EMA50_Bias"
timeframe = "12h"
leverage = 1.0