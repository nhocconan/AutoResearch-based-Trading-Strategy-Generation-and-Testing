#!/usr/bin/env python3
"""
1h_4d_1d_RSI_Divergence_v1
Hypothesis: Use RSI divergence between 1h price and 4h/1d RSI to identify reversals.
Long when 1h price makes lower low but 4h RSI makes higher low (bullish divergence).
Short when 1h price makes higher high but 1d RSI makes lower high (bearish divergence).
Designed for low trade frequency (target: 60-150 total over 4 years) to minimize fee drag.
Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_1d_RSI_Divergence_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h RSI for entry confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1h = (100 - (100 / (1 + rs))).fillna(50).values
    
    # 4h RSI for divergence (higher timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    delta_4h = pd.Series(close_4h).diff()
    gain_4h = delta_4h.clip(lower=0)
    loss_4h = -delta_4h.clip(upper=0)
    avg_gain_4h = gain_4h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_4h = avg_gain_4h / avg_loss_4h.replace(0, np.nan)
    rsi_4h = (100 - (100 / (1 + rs_4h))).fillna(50).values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1d RSI for divergence (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.clip(lower=0)
    loss_1d = -delta_1d.clip(upper=0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d.replace(0, np.nan)
    rsi_1d = (100 - (100 / (1 + rs_1d))).fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Lookback for divergence detection
    lookback = 10
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Need enough lookback
        if i - lookback < 0:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Bullish divergence: price lower low, RSI higher low
        price_low_1h = np.min(low[i-lookback:i+1])
        price_low_prev = np.min(low[i-lookback*2:i-lookback])
        rsi_low_4h = np.min(rsi_4h_aligned[i-lookback:i+1])
        rsi_low_prev_4h = np.min(rsi_4h_aligned[i-lookback*2:i-lookback])
        rsi_low_1d = np.min(rsi_1d_aligned[i-lookback:i+1])
        rsi_low_prev_1d = np.min(rsi_1d_aligned[i-lookback*2:i-lookback])
        
        bullish_div = (price_low_1h < price_low_prev and 
                      rsi_low_4h > rsi_low_prev_4h and
                      rsi_low_1d > rsi_low_prev_1d)
        
        # Bearish divergence: price higher high, RSI lower high
        price_high_1h = np.max(high[i-lookback:i+1])
        price_high_prev = np.max(high[i-lookback*2:i-lookback])
        rsi_high_4h = np.max(rsi_4h_aligned[i-lookback:i+1])
        rsi_high_prev_4h = np.max(rsi_4h_aligned[i-lookback*2:i-lookback])
        rsi_high_1d = np.max(rsi_1d_aligned[i-lookback:i+1])
        rsi_high_prev_1d = np.max(rsi_1d_aligned[i-lookback*2:i-lookback])
        
        bearish_div = (price_high_1h > price_high_prev and
                      rsi_high_4h < rsi_high_prev_4h and
                      rsi_high_1d < rsi_high_prev_1d)
        
        # Entry conditions with RSI extremes
        rsi_oversold = rsi_1h[i] < 30
        rsi_overbought = rsi_1h[i] > 70
        
        if bullish_div and rsi_oversold and position != 1:
            position = 1
            signals[i] = 0.20
        elif bearish_div and rsi_overbought and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and (rsi_1h[i] > 50 or not in_session[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_1h[i] < 50 or not in_session[i]):
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals