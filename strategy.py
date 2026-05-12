#!/usr/bin/env python3
"""
4H_RSI_DIVERGENCE_VOLUME_CONFIRMATION
Hypothesis: Use RSI divergence (bullish/bearish) with volume confirmation and 1d trend filter to capture reversals in both bull and bear markets.
RSI divergence signals exhaustion of momentum, and when confirmed by volume and trend, provides high-probability reversals.
Trades only on clear divergences to keep trade frequency low (target: 20-40/year).
Works in bull markets (catching pullbacks in uptrend) and bear markets (catching rallies in downtrend).
"""
name = "4H_RSI_DIVERGENCE_VOLUME_CONFIRMATION"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Detect RSI divergence (bullish and bearish)
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    lookback = 10  # lookback period for divergence
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Bullish divergence
        if low[i] < low[i-lookback] and rsi[i] > rsi[i-lookback]:
            # Check if RSI is making a higher low
            min_rsi_idx = np.argmin(rsi[i-lookback:i+1])
            min_price_idx = np.argmin(low[i-lookback:i+1])
            if min_rsi_idx > min_price_idx:  # RSI low occurs after price low
                bullish_div[i] = True
        # Bearish divergence
        if high[i] > high[i-lookback] and rsi[i] < rsi[i-lookback]:
            # Check if RSI is making a lower high
            max_rsi_idx = np.argmax(rsi[i-lookback:i+1])
            max_price_idx = np.argmax(high[i-lookback:i+1])
            if max_rsi_idx < max_price_idx:  # RSI high occurs before price high
                bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # LONG: Bullish divergence + volume spike + above 1d EMA50 (uptrend context)
            if bullish_div[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # SHORT: Bearish divergence + volume spike + below 1d EMA50 (downtrend context)
            elif bearish_div[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>=70) or trend reversal
            if rsi[i] >= 70 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<=30) or trend reversal
            if rsi[i] <= 30 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals