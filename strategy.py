#!/usr/bin/env python3
"""
4h_rsi_pullback_1d_ema_trend_v1
Hypothesis: On 4h timeframe, enter long when price pulls back to EMA20 during uptrend (EMA50 > EMA200 on 1d) with RSI < 40, short when price rallies to EMA20 during downtrend (EMA50 < EMA200 on 1d) with RSI > 60. Use 1d trend filter to avoid counter-trend trades. Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag while capturing meaningful swings.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_1d_ema_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    
    # Trend: 1 = uptrend (EMA50 > EMA200), -1 = downtrend (EMA50 < EMA200), 0 = sideways
    ema_trend = np.where(ema50_1d > ema200_1d, 1, np.where(ema50_1d < ema200_1d, -1, 0))
    ema_trend_aligned = align_htf_to_ltf(prices, df_1d, ema_trend)
    
    # Calculate 4h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 4h EMA20 for pullback
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if data not available
        if (np.isnan(ema_trend_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(ema20[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        trend = ema_trend_aligned[i]
        rsi_val = rsi_values[i]
        price = close[i]
        ema20_val = ema20[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 60 or trend turns down
            if rsi_val > 60 or trend == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 40 or trend turns up
            if rsi_val < 40 or trend == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if trend == 1:  # Uptrend - look for long pullback
                if rsi_val < 40 and price <= ema20_val * 1.005:  # Near EMA20
                    position = 1
                    signals[i] = 0.25
            elif trend == -1:  # Downtrend - look for short rally
                if rsi_val > 60 and price >= ema20_val * 0.995:  # Near EMA20
                    position = -1
                    signals[i] = -0.25
    
    return signals