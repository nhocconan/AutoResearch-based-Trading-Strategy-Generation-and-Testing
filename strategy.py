#!/usr/bin/env python3
"""
6h_rsi_divergence_1d_trend_volume_v1
Hypothesis: On 6h timeframe, detect RSI divergence (bullish/bearish) with price 
to anticipate reversals. Use 1d EMA200 as trend filter to only take divergences 
in direction of higher timeframe trend. Volume confirmation ensures momentum 
behind the move. Works in bull markets (buy bullish div in uptrend) and bear 
markets (sell bearish div in downtrend). Targets 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi_divergence_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema200_6h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 6h RSI(14) for divergence detection
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = np.zeros_like(close_prices)
        rsi_vals[period:] = 100 - (100 / (1 + rs[period:]))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # 20-period volume average on 6h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(rsi_vals[i]) or np.isnan(ema200_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        # Check for RSI divergence over last 10 periods
        lookback = 10
        if i < lookback:
            signals[i] = 0.0
            continue
            
        # Bullish divergence: price makes lower low, RSI makes higher low
        # Bearish divergence: price makes higher high, RSI makes lower high
        price_low = low[i-lookback:i+1]
        price_high = high[i-lookback:i+1]
        rsi_low = rsi_vals[i-lookback:i+1]
        rsi_high = rsi_vals[i-lookback:i+1]
        
        # Find local minima and maxima
        price_min_idx = np.argmin(price_low)
        price_max_idx = np.argmax(price_high)
        rsi_min_idx = np.argmin(rsi_low)
        rsi_max_idx = np.argmax(rsi_high)
        
        bullish_div = (price_min_idx == lookback and rsi_min_idx != lookback and 
                      rsi_vals[i] > rsi_vals[i-lookback] and 
                      close[i] < close[i-lookback])
        bearish_div = (price_max_idx == lookback and rsi_max_idx != lookback and 
                      rsi_vals[i] < rsi_vals[i-lookback] and 
                      close[i] > close[i-lookback])
        
        if position == 1:  # Long position
            # Exit: bearish divergence or price breaks below EMA200
            if bearish_div or close[i] < ema200_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: bullish divergence or price breaks above EMA200
            if bullish_div or close[i] > ema200_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish divergence long in uptrend (price > EMA200)
            if (bullish_div and 
                vol_confirm and 
                close[i] > ema200_6h[i]):
                position = 1
                signals[i] = 0.25
            # Bearish divergence short in downtrend (price < EMA200)
            elif (bearish_div and 
                  vol_confirm and 
                  close[i] < ema200_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals