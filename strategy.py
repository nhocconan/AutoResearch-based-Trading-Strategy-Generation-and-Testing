#!/usr/bin/env python3
"""
1d RSI Reversal with Weekly Trend Filter and Volume Confirmation
Hypothesis: RSI extremes on daily timeframe combined with weekly trend direction
provide high-probability reversal entries in both bull and bear markets.
Volume confirmation filters out weak moves. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume filter: current volume > 1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        vol_ok = vol_filter[i]
        weekly_trend = ema50_1w_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: RSI oversold in uptrend (weekly)
            if rsi_val < 30 and vol_ok and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in downtrend (weekly)
            elif rsi_val > 70 and vol_ok and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral or trend changes
            if rsi_val > 50 or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI returns to neutral or trend changes
            if rsi_val < 50 or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_Reversal_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0