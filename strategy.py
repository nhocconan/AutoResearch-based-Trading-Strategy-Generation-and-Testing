#!/usr/bin/env python3
# 4h_1d_rsi_divergence_volume_reversal_v1
# Hypothesis: 4-hour RSI divergence (bullish/bearish) with volume confirmation and 1-day EMA trend filter.
# Bullish divergence: price makes lower low but RSI makes higher low, with volume confirmation.
# Bearish divergence: price makes higher high but RSI makes lower high, with volume confirmation.
# Works in both bull and bear markets by capturing exhaustion points and reversals.
# Uses 1-day EMA50 to filter trades in direction of higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_rsi_divergence_volume_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi = np.zeros(n)
    
    # Initialize first average
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # 20-period average volume for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d_50[i] = close_1d[i] * (2/51) + ema_1d_50[i-1] * (49/51)
    
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after RSI and volume warmup
        price = close[i]
        curr_low = low[i]
        curr_high = high[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        curr_rsi = rsi[i]
        ema_1d = ema_1d_50_aligned[i]
        
        if np.isnan(avg_vol) or np.isnan(ema_1d) or np.isnan(curr_rsi):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.5 * avg_vol
        
        # Check for bullish divergence: lower low in price, higher low in RSI
        bullish_div = False
        if i >= 5:  # Need at least 5 periods to compare
            # Find recent low in price and RSI
            lookback = 5
            price_lows = []
            rsi_lows = []
            for j in range(i-lookback, i+1):
                if j > 0 and low[j] <= low[j-1] and low[j] <= low[j+1] if j < n-1 else True:
                    price_lows.append((j, low[j]))
                    rsi_lows.append((j, rsi[j]))
            
            if len(price_lows) >= 2:
                # Check if last two lows show divergence
                last_low_price = price_lows[-1][1]
                prev_low_price = price_lows[-2][1]
                last_low_rsi = rsi_lows[-1][1]
                prev_low_rsi = rsi_lows[-2][1]
                
                if last_low_price < prev_low_price and last_low_rsi > prev_low_rsi:
                    bullish_div = True
        
        # Check for bearish divergence: higher high in price, lower high in RSI
        bearish_div = False
        if i >= 5:
            lookback = 5
            price_highs = []
            rsi_highs = []
            for j in range(i-lookback, i+1):
                if j > 0 and high[j] >= high[j-1] and high[j] >= high[j+1] if j < n-1 else True:
                    price_highs.append((j, high[j]))
                    rsi_highs.append((j, rsi[j]))
            
            if len(price_highs) >= 2:
                last_high_price = price_highs[-1][1]
                prev_high_price = price_highs[-2][1]
                last_high_rsi = rsi_highs[-1][1]
                prev_high_rsi = rsi_highs[-2][1]
                
                if last_high_price > prev_high_price and last_high_rsi < prev_high_rsi:
                    bearish_div = True
        
        if position == 1:  # Long position
            # Exit on bearish divergence or price breaks below recent low with volume
            if bearish_div and vol_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on bullish divergence or price breaks above recent high with volume
            if bullish_div and vol_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on bullish divergence with volume and price above 1-day EMA
            if bullish_div and vol_surge and price > ema_1d:
                position = 1
                signals[i] = 0.25
            # Enter short on bearish divergence with volume and price below 1-day EMA
            elif bearish_div and vol_surge and price < ema_1d:
                position = -1
                signals[i] = -0.25
    
    return signals