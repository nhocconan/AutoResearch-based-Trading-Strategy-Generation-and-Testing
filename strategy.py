#!/usr/bin/env python3
"""
1h RSI Pullback with 4h Trend and 1d Volume Confirmation v1
Hypothesis: In strong trends (4h EMA50), pullbacks to RSI(14) < 30 (long) or >70 (short) on 1h offer high-probability entries. Volume confirmation from 1d average volume > 1.5x filters low-conviction moves. Session filter (08-20 UTC) reduces noise. Targets 15-37 trades/year by requiring multi-timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h_trend_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = df_4h['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_avg_1d = df_1d['volume'].ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 or trend turns bearish
            if rsi[i] >= 70 or close[i] < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 or trend turns bullish
            if rsi[i] <= 30 or close[i] > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: RSI oversold in uptrend with volume confirmation
            if (rsi[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > vol_avg_1d_aligned[i] * 1.5):
                position = 1
                signals[i] = 0.20
            # Short: RSI overbought in downtrend with volume confirmation
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > vol_avg_1d_aligned[i] * 1.5):
                position = -1
                signals[i] = -0.20
    
    return signals