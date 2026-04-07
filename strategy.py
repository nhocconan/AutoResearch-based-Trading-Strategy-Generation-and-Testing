#!/usr/bin/env python3
"""
1h_rsi_pullback_4h1d_trend_volume_v1
Hypothesis: RSI pullbacks in direction of 4h/1d trend with volume confirmation. 
In bull markets: buy RSI<40 pullbacks in uptrend. In bear markets: sell RSI>60 bounces in downtrend.
Uses 4h EMA50 for trend, 1d EMA200 for filter, volume >1.5x average for confirmation.
Target: 15-30 trades/year (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h1d_trend_volume_v1"
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
    
    # 4h data for trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_4h = df_4h['close'].ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    ema_1d = df_1d['close'].ewm(span=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation (24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 60 or trend turns bearish
            if rsi[i] > 60 or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI < 40 or trend turns bullish
            if rsi[i] < 40 or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: RSI < 40 pullback with volume and bullish trend (4h & 1d)
            if (rsi[i] < 40 and vol_confirm and 
                close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI > 60 bounce with volume and bearish trend (4h & 1d)
            elif (rsi[i] > 60 and vol_confirm and 
                  close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals