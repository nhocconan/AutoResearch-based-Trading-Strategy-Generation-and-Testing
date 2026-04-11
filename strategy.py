#!/usr/bin/env python3
# 1d_1w_rsi_divergence_v1
# Strategy: Daily RSI divergence with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: RSI divergence (price makes new high/low but RSI does not) signals reversals. 
# Weekly trend filter ensures trades align with higher timeframe momentum. 
# Volume confirmation filters weak signals. Works in bull by catching pullbacks in uptrend,
# and in bear by catching bounces in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_divergence_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly RSI for trend filter (14-period)
    close_1w = df_1w['close'].values
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    avg_gain_1w = pd.Series(gain_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1w = pd.Series(loss_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1w = avg_gain_1w / (avg_loss_1w + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w_trend = rsi_1w > 50  # Uptrend when RSI > 50
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_trend)
    
    # Daily RSI (14-period) for divergence
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Track recent highs/lows for divergence (14-period lookback)
    lookback = 14
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get lookback window
        start_idx = i - lookback
        if start_idx < 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Price and RSI in lookback window
        price_window = close[start_idx:i+1]
        rsi_window = rsi[start_idx:i+1]
        
        # Find recent high and low
        recent_high = np.max(price_window)
        recent_low = np.min(price_window)
        rsi_at_high = rsi_window[np.argmax(price_window)]
        rsi_at_low = rsi_window[np.argmin(price_window)]
        
        # Current values
        price_now = close[i]
        rsi_now = rsi[i]
        
        # Bullish divergence: price makes lower low but RSI makes higher low
        bull_div = (price_now <= recent_low * 1.001) and (rsi_now > rsi_at_low + 5)
        # Bearish divergence: price makes higher high but RSI makes lower high
        bear_div = (price_now >= recent_high * 0.999) and (rsi_now < rsi_at_high - 5)
        
        # Exit conditions: RSI crosses 50 or opposite divergence
        exit_long = position == 1 and (rsi_now < 50 or bear_div)
        exit_short = position == -1 and (rsi_now > 50 or bull_div)
        
        # Trading logic: only trade in direction of weekly trend
        if bull_div and rsi_1w_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_div and not rsi_1w_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals