#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h/1d trend filter and session filter (08-20 UTC)
# RSI < 20/ > 80 for oversold/overbought entries with 4h trend alignment
# 1d trend filter avoids counter-trend trades in strong trends
# Session filter reduces noise during low-liquidity hours
# Designed for mean reversion in ranging markets while avoiding strong trends
name = "1h_RSI_MeanReversion_4hTrend_1dFilter_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA20 trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: RSI < 20 (oversold) and price above both 4h and 1d EMA (uptrend)
            if rsi_val < 20 and price > ema_20_4h_aligned[i] and price > ema_50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 80 (overbought) and price below both 4h and 1d EMA (downtrend)
            elif rsi_val > 80 and price < ema_20_4h_aligned[i] and price < ema_50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion) or price below 4h EMA (trend change)
            if rsi_val > 50 or price < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion) or price above 4h EMA (trend change)
            if rsi_val < 50 or price > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals