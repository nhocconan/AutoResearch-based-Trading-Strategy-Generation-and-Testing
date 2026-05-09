# 6h_RSI_Divergence_Trend_Follow
# Hypothesis: Combines RSI divergence detection with daily trend filter and volume confirmation.
# RSI divergence signals potential reversals, while daily trend filter ensures we trade with higher timeframe momentum.
# Volume spike confirms institutional participation. Designed for low trade frequency (<30/year) to minimize fee drag.
# Works in bull markets (trend continuation on pullbacks) and bear markets (mean reversion at extremes via divergence).

name = "6h_RSI_Divergence_Trend_Follow"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 14-period RSI on 6h
    if n < 14:
        return np.zeros(n)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi = np.full(n, 50.0)  # Initialize with neutral RSI
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Detect RSI divergence (lookback 5 periods)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    lookback = 5
    for i in range(lookback, n):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if low[i] < low[i-lookback] and rsi[i] > rsi[i-lookback]:
            # Check if it's a meaningful divergence
            if low[i] == np.min(low[i-lookback:i+1]) and rsi[i] == np.max(rsi[i-lookback:i+1]):
                bullish_div[i] = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if high[i] > high[i-lookback] and rsi[i] < rsi[i-lookback]:
            # Check if it's a meaningful divergence
            if high[i] == np.max(high[i-lookback:i+1]) and rsi[i] == np.min(rsi[i-lookback:i+1]):
                bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, lookback)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema_20_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_20_1d_aligned[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Bullish RSI divergence AND price > 1d EMA20 (uptrend) AND volume > 1.8x average
            if bullish_div[i] and close[i] > ema_1d and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish RSI divergence AND price < 1d EMA20 (downtrend) AND volume > 1.8x average
            elif bearish_div[i] and close[i] < ema_1d and vol > 1.8 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish RSI divergence OR trend reverses (price < 1d EMA20)
            if bearish_div[i] or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish RSI divergence OR trend reverses (price > 1d EMA20)
            if bullish_div[i] or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf