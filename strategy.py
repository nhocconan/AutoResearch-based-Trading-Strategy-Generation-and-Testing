# 1h_RSI_SMA_Trend_Filter_v1
# Hypothesis: Use 1h for entry timing with RSI pullback into SMA50 during strong 4h/1d trends.
# In bull markets: 4h/1d trend up + RSI pullback to SMA50 = long opportunity.
# In bear markets: 4h/1d trend down + RSI pullback to SMA50 = short opportunity.
# Session filter (08-20 UTC) reduces noise. Target 15-37 trades/year.
# Risk: Exit on opposite RSI extreme or trend reversal.

#!/usr/bin/env python3
name = "1h_RSI_SMA_Trend_Filter_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d trend filter: EMA100
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h SMA50
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # need 1d EMA100 and 1h SMA50
    
    for i in range(start_idx, n):
        if not session_mask[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_100_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(sma_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h/1d uptrend + RSI pullback to SMA50 (RSI < 50 and closing above SMA50)
            if (close[i] > ema_50_4h_aligned[i] and 
                close[i] > ema_100_1d_aligned[i] and 
                rsi[i] < 50 and 
                close[i] > sma_50[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h/1d downtrend + RSI pullback to SMA50 (RSI > 50 and closing below SMA50)
            elif (close[i] < ema_50_4h_aligned[i] and 
                  close[i] < ema_100_1d_aligned[i] and 
                  rsi[i] > 50 and 
                  close[i] < sma_50[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: trend reversal or RSI overbought
            if (close[i] < ema_50_4h_aligned[i] or 
                close[i] < ema_100_1d_aligned[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: trend reversal or RSI oversold
            if (close[i] > ema_50_4h_aligned[i] or 
                close[i] > ema_100_1d_aligned[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals