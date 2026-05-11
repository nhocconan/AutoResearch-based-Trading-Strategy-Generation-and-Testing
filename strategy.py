#!/usr/bin/env python3
"""
12h_RSI_Divergence_4hTrend_Filter
Hypothesis: Use 12h RSI divergence (bullish/bearish) as entry signal, filtered by 4h EMA trend direction.
Trades only in direction of 4h trend to avoid counter-trend whipsaws. RSI divergence signals reversals
with higher probability in ranging markets, while trend filter ensures alignment with higher timeframe momentum.
Designed for low trade frequency (~20-40/year) to minimize fee drag. Works in bull/bear by following 4h trend.
"""

name = "12h_RSI_Divergence_4hTrend_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # === 12h Data for RSI and Divergence Detection ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:  # RSI needs at least 14 periods
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h RSI(14)
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h[:13] = np.nan  # Not enough data for first 13 periods
    
    # === 4h Data for Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Align 12h RSI to lower timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Detect RSI divergence: bullish (price makes lower low, RSI makes higher low)
    # and bearish (price makes higher high, RSI makes lower high)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers RSI calculation)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(ema20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Check for bullish divergence: price makes lower low, RSI makes higher low
            # Look back 3 periods for divergence confirmation
            if i >= 3:
                price_lower_low = close[i] < close[i-1] and close[i-1] < close[i-2]
                rsi_higher_low = (not np.isnan(rsi_12h_aligned[i]) and 
                                not np.isnan(rsi_12h_aligned[i-1]) and 
                                not np.isnan(rsi_12h_aligned[i-2]) and
                                rsi_12h_aligned[i] > rsi_12h_aligned[i-1] and 
                                rsi_12h_aligned[i-1] > rsi_12h_aligned[i-2])
                
                # Check for bearish divergence: price makes higher high, RSI makes lower high
                price_higher_high = close[i] > close[i-1] and close[i-1] > close[i-2]
                rsi_lower_high = (not np.isnan(rsi_12h_aligned[i]) and 
                                not np.isnan(rsi_12h_aligned[i-1]) and 
                                not np.isnan(rsi_12h_aligned[i-2]) and
                                rsi_12h_aligned[i] < rsi_12h_aligned[i-1] and 
                                rsi_12h_aligned[i-1] < rsi_12h_aligned[i-2])
                
                # Bullish entry: bullish divergence + price above 4h EMA (uptrend)
                if price_lower_low and rsi_higher_low and close[i] > ema20_4h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: bearish divergence + price below 4h EMA (downtrend)
                elif price_higher_high and rsi_lower_high and close[i] < ema20_4h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: RSI becomes overbought or trend turns down
            if (rsi_12h_aligned[i] > 70 or close[i] < ema20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        
        elif position == -1:
            # Short exit: RSI becomes oversold or trend turns up
            if (rsi_12h_aligned[i] < 30 or close[i] > ema20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals