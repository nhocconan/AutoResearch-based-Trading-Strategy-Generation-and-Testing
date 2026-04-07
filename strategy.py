#!/usr/bin/env python3
"""
1h_rsi_mean_reversion_4h1d_trend_volume_v1
Hypothesis: RSI mean reversion on 1h timeframe with 4h/1d trend filter and volume confirmation.
In bull markets: buy oversold RSI during uptrends. In bear markets: sell overbought RSI during downtrends.
Volume filter ensures institutional participation. Targets 15-35 trades/year with strict entry conditions.
Works in both bull and bear markets by trading in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_mean_reversion_4h1d_trend_volume_v1"
timeframe = "1h"
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
    
    # 4h EMA200 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema200_4h = pd.Series(df_4h['close'].values).ewm(span=200, adjust=False).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 14-period RSI on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                position = 0
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Trend alignment: both 4h and 1d EMAs agree
        uptrend = close[i] > ema200_4h_aligned[i] and close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema200_4h_aligned[i] and close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI overbought OR trend breaks down
            if rsi[i] >= 70 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI oversold OR trend breaks up
            if rsi[i] <= 30 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: RSI oversold + volume confirmation + uptrend
            if (rsi[i] <= 30 and 
                vol_confirm and 
                uptrend):
                position = 1
                signals[i] = 0.20
            # Short: RSI overbought + volume confirmation + downtrend
            elif (rsi[i] >= 70 and 
                  vol_confirm and 
                  downtrend):
                position = -1
                signals[i] = -0.20
    
    return signals