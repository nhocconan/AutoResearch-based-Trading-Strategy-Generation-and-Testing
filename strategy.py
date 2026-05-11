#!/usr/bin/env python3
"""
1h_4h1d_Trend_Divergence_RSI
Hypothesis: In 1h timeframe, take counter-trend positions when 4h RSI shows extreme
overbought/oversold (>70 or <30) but 1h price is near 20-period EMA, indicating
short-term exhaustion within larger trend. Use 1d trend filter (price > EMA50 for long,
price < EMA50 for short) to avoid counter-trend trades in strong trends. Designed
for low frequency (15-30 trades/year) to work in both bull (sell rallies in uptrend)
and bear (buy dips in downtrend) markets by fading short-term extremes while
respecting intermediate trend.
"""

name = "1h_4h1d_Trend_Divergence_RSI"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for RSI
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- 4h RSI(14) ---
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_4h = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # --- 1d EMA50 for trend filter ---
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- 1h EMA20 for entry timing ---
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # --- Session filter: 08-20 UTC ---
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_20[i])):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Skip outside session
        if not in_session[i]:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: 4h RSI < 30 (oversold) AND price near 1h EMA20 AND above 1d EMA50 (uptrend)
        # Short: 4h RSI > 70 (overbought) AND price near 1h EMA20 AND below 1d EMA50 (downtrend)
        price_near_ema = np.abs(close[i] - ema_20[i]) / ema_20[i] < 0.005  # Within 0.5%
        
        long_entry = (rsi_4h_aligned[i] < 30) and \
                     price_near_ema and \
                     (close[i] > ema_50_1d_aligned[i])
        
        short_entry = (rsi_4h_aligned[i] > 70) and \
                      price_near_ema and \
                      (close[i] < ema_50_1d_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions:
            # Long: 4h RSI > 50 (momentum shift) OR price deviates from EMA20 OR below 1d EMA50
            # Short: 4h RSI < 50 OR price deviates from EMA20 OR above 1d EMA50
            if position == 1:
                if (rsi_4h_aligned[i] > 50) or \
                   (np.abs(close[i] - ema_20[i]) / ema_20[i] > 0.01) or \
                   (close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                if (rsi_4h_aligned[i] < 50) or \
                   (np.abs(close[i] - ema_20[i]) / ema_20[i] > 0.01) or \
                   (close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals