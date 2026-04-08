#!/usr/bin/env python3
# 1h_rsi_reversion_4h1d_trend_v1
# Hypothesis: Mean reversion on 1h RSI extremes filtered by 4h/1d trend. Goes long when 1h RSI < 30 and price > 4h EMA50 and 1d EMA200 (uptrend). Short when RSI > 70 and price < 4h EMA50 and 1d EMA200 (downtrend). Uses volume confirmation to avoid false signals. Target: 20-50 trades/year to minimize fee drag. Works in bull (trend follow pullbacks) and bear (mean reversion in range).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_reversion_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 4h EMA50 trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA200 trend
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1h RSI(14)
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
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filters
        uptrend_4h = close[i] > ema50_4h_aligned[i]
        uptrend_1d = close[i] > ema200_1d_aligned[i]
        downtrend_4h = close[i] < ema50_4h_aligned[i]
        downtrend_1d = close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 50 or trend breakdown
            if rsi[i] > 50 or not (uptrend_4h and uptrend_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 or trend breakdown
            if rsi[i] < 50 or not (downtrend_4h and downtrend_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long entry: RSI oversold in uptrend
                if rsi[i] < 30 and uptrend_4h and uptrend_1d:
                    position = 1
                    signals[i] = 0.20
                # Short entry: RSI overbought in downtrend
                elif rsi[i] > 70 and downtrend_4h and downtrend_1d:
                    position = -1
                    signals[i] = -0.20
    
    return signals