#!/usr/bin/env python3
# 6h_funding_rate_mean_reversion_v1
# Hypothesis: Funding rate mean reversion works on BTC/ETH/SOL perpetuals. 
# Extreme funding rates (>0.05% or <-0.05%) predict mean reversion in next funding period.
# Uses 8h funding rate data as HTF proxy, combined with 6h price action for entry timing.
# Works in both bull and bear markets as funding extremes occur in all regimes.
# Target: 60-120 trades over 4 years (15-30/year) with 0.25 position size.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_funding_rate_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Get funding rate data (using 8h as proxy since we don't have direct funding)
    # We'll simulate funding based on price deviation from 8h VWAP
    df_8h = get_htf_data(prices, '8h')
    if len(df_8h) < 20:
        return np.zeros(n)
    
    # Calculate 8h VWAP approximation
    typical_price_8h = (df_8h['high'].values + df_8h['low'].values + df_8h['close'].values) / 3
    volume_8h = df_8h['volume'].values
    
    vwap_sum = np.cumsum(typical_price_8h * volume_8h)
    vol_sum = np.cumsum(volume_8h)
    vwap_8h = np.where(vol_sum > 0, vwap_sum / vol_sum, typical_price_8h)
    
    # Calculate funding rate proxy: deviation from VWAP
    # Positive when price > VWAP (longs pay shorts)
    funding_proxy = (typical_price_8h - vwap_8h) / vwap_8h
    
    # Smooth funding proxy to reduce noise
    funding_smooth = np.zeros_like(funding_proxy)
    alpha = 2 / (8 + 1)  # 8-period EMA
    funding_smooth[0] = funding_proxy[0]
    for i in range(1, len(funding_proxy)):
        funding_smooth[i] = alpha * funding_proxy[i] + (1 - alpha) * funding_smooth[i-1]
    
    # Align funding rate to 6h timeframe
    funding_6h = align_htf_to_ltf(prices, df_8h, funding_smooth)
    
    # Bollinger Bands on 6h for volatility context
    sma_20 = np.zeros(n)
    std_20 = np.zeros(n)
    for i in range(20, n):
        sma_20[i] = np.mean(close[i-20:i])
        std_20[i] = np.std(close[i-20:i])
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(funding_6h[i]) or 
            np.isnan(sma_20[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: funding becomes extremely negative or RSI overbought
            if funding_6h[i] < -0.0005 or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: funding becomes extremely positive or RSI oversold
            if funding_6h[i] > 0.0005 or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: extremely negative funding + price near lower BB + RSI oversold
            if (funding_6h[i] < -0.0003 and 
                close[i] <= lower_bb[i] * 1.01 and  # Near lower BB
                rsi[i] < 35):
                position = 1
                signals[i] = 0.25
            # Enter short: extremely positive funding + price near upper BB + RSI overbought
            elif (funding_6h[i] > 0.0003 and 
                  close[i] >= upper_bb[i] * 0.99 and  # Near upper BB
                  rsi[i] > 65):
                position = -1
                signals[i] = -0.25
    
    return signals