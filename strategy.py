#!/usr/bin/env python3
"""
1h_RSI_Divergence_4hTrend_SessionFilter_v1
Hypothesis: Trade 1h RSI bullish/bearish divergences aligned with 4h EMA50 trend during active UTC 08-20 session.
RSI divergence identifies potential reversals with confluence from higher timeframe trend filter.
Session filter reduces noise trades by focusing on liquid hours. Discrete sizing 0.20 limits fee drag.
Target: 15-37 trades/year to avoid fee drag while working in both bull (trend continuation) and bear (mean reversion) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for RSI and 4h EMA50
    start_idx = max(14, 50)
    
    for i in range(start_idx, n):
        # Skip if outside session or data not ready
        if not in_session[i] or np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend from EMA50
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)[i]
        if np.isnan(close_4h_aligned):
            signals[i] = 0.0
            continue
            
        if close_4h_aligned > ema_50_4h_aligned[i]:
            htf_trend = 'bullish'  # favor longs
        elif close_4h_aligned < ema_50_4h_aligned[i]:
            htf_trend = 'bearish'  # favor shorts
        else:
            htf_trend = 'neutral'  # no strong trend
        
        # Detect RSI divergence (simple peak/trough comparison)
        lookback = 10
        if i < lookback:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Find local peaks and troughs in RSI and price
        rsi_window = rsi[i-lookback:i+1]
        price_window = close[i-lookback:i+1]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if len(rsi_window) >= 3 and len(price_window) >= 3:
            price_min_idx = np.argmin(price_window)
            rsi_min_idx = np.argmin(rsi_window)
            if price_min_idx > 0 and rsi_min_idx > 0 and price_min_idx < lookback and rsi_min_idx < lookback:
                # Check if current point is a potential low
                if i == lookback and price_window[-1] == np.min(price_window) and rsi_window[-1] > np.min(rsi_window[:-1]):
                    bullish_div = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if len(rsi_window) >= 3 and len(price_window) >= 3:
            price_max_idx = np.argmax(price_window)
            rsi_max_idx = np.argmax(rsi_window)
            if price_max_idx > 0 and rsi_max_idx > 0 and price_max_idx < lookback and rsi_max_idx < lookback:
                # Check if current point is a potential high
                if i == lookback and price_window[-1] == np.max(price_window) and rsi_window[-1] < np.max(rsi_window[:-1]):
                    bearish_div = True
        
        if position == 0:
            # Long setup: bullish divergence AND (bullish or neutral 4h trend)
            long_setup = bullish_div and (htf_trend in ['bullish', 'neutral'])
            
            # Short setup: bearish divergence AND (bearish or neutral 4h trend)
            short_setup = bearish_div and (htf_trend in ['bearish', 'neutral'])
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: bearish divergence OR strong bearish 4h trend
            if bearish_div or htf_trend == 'bearish':
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: bullish divergence OR strong bullish 4h trend
            if bullish_div or htf_trend == 'bullish':
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_RSI_Divergence_4hTrend_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0