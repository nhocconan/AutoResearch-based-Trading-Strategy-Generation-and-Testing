#!/usr/bin/env python3
"""
4h RSI Divergence + Volume Confirmation with 1d Trend Filter
Hypothesis: RSI divergences signal exhaustion in overbought/oversold conditions. 
Combined with volume confirmation and 1d EMA trend filter, this captures reversals 
with controlled frequency. Works in both bull (buy oversold dips) and bear (sell overbought rallies).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Price and RSI rolling extremes for divergence detection
    lookback = 14
    price_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    price_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    rsi_high = pd.Series(rsi).rolling(window=lookback, min_periods=lookback).max().values
    rsi_low = pd.Series(rsi).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        trend = ema50_1d_aligned[i]
        rsi_val = rsi[i]
        vol_ok = vol_spike[i]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bull_div = (low[i] <= price_low[i] * 1.001) and (rsi_val >= rsi_low[i] * 1.02)
        # Bearish divergence: price makes higher high, RSI makes lower high
        bear_div = (high[i] >= price_high[i] * 0.999) and (rsi_val <= rsi_high[i] * 0.98)
        
        if position == 0:
            # Enter long on bullish divergence + volume + uptrend filter
            if bull_div and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish divergence + volume + downtrend filter
            elif bear_div and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on RSI overbought or trend failure
            if rsi_val > 70 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on RSI oversold or trend failure
            if rsi_val < 30 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Divergence_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0