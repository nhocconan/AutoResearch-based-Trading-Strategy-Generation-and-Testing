#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI Pullback with 4h/1d Trend Filter
# Hypothesis: In strong trends (4h EMA50 > EMA200 and 1d close > SMA50), 
# RSI pullbacks on 1f provide high-probability entries with low drawdown.
# Uses 4h/1d for trend direction, 1h for timing. Session filter (08-20 UTC) reduces noise.
# Target: 15-35 trades/year (60-140 total) to minimize fee drag.

name = "1h_rsi_pullback_4h1d_trend_v1"
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
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # 1h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(150, n):  # warmup for 200 EMA
        # Skip if required data not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_4h_aligned[i]) or 
            np.isnan(sma_50_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Trend determination
        uptrend_4h = ema_50_4h_aligned[i] > ema_200_4h_aligned[i]
        uptrend_1d = close[i] > sma_50_1d_aligned[i]
        
        downtrend_4h = ema_50_4h_aligned[i] < ema_200_4h_aligned[i]
        downtrend_1d = close[i] < sma_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI overbought or trend fails
            if rsi[i] >= 70 or not (uptrend_4h and uptrend_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: RSI oversold or trend fails
            if rsi[i] <= 30 or not (downtrend_4h and downtrend_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Long: uptrend on both timeframes + RSI pullback (oversold)
            if uptrend_4h and uptrend_1d and rsi[i] <= 30:
                position = 1
                signals[i] = 0.20
            # Short: downtrend on both timeframes + RSI pullback (overbought)
            elif downtrend_4h and downtrend_1d and rsi[i] >= 70:
                position = -1
                signals[i] = -0.20
    
    return signals