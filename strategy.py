#!/usr/bin/env python3
# 12h_rsi_trend_volume_v1
# Hypothesis: Uses RSI(14) extremes with 1d trend filter and volume confirmation.
# Goes long when RSI < 30 in uptrend (price > 1d EMA50) with volume surge.
# Goes short when RSI > 70 in downtrend (price < 1d EMA50) with volume surge.
# Designed for low trade frequency (12-37/year) to avoid fee drift, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_rsi_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 14:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 14
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # RSI signals
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 50 or trend change
            if rsi[i] > 50 or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 or trend change
            if rsi[i] < 50 or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: RSI oversold in uptrend
                if daily_uptrend and rsi_oversold:
                    position = 1
                    signals[i] = 0.25
                # Short entry: RSI overbought in downtrend
                elif daily_downtrend and rsi_overbought:
                    position = -1
                    signals[i] = -0.25
    
    return signals