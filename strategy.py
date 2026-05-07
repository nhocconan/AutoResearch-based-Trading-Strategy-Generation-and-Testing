#!/usr/bin/env python3
# 6h_River_Pullback_200EMA_RSI
# Hypothesis: Buy dips in uptrends (price > 200EMA) when RSI pulls back from oversold (RSI < 30) on 6h chart.
# Sell bounces in downtrends (price < 200EMA) when RSI pulls back from overbought (RSI > 70) on 6h chart.
# Trend filter uses 12h 200EMA for higher timeframe confirmation to avoid counter-trend trades.
# Volume confirmation requires 1.2x average volume to ensure participation.
# Designed for low trade frequency (~20-40/year) to minimize fee drag while capturing meaningful swings.

timeframe = "6h"
name = "6h_River_Pullback_200EMA_RSI"
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
    
    # Get 12h data for higher timeframe trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate 12h 200EMA for trend filter
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate 6h 200EMA for trend definition
    ema_200_6h = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: 1.2x average volume (4-period = 1 day on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 14, 4)  # Ensure we have EMA, RSI, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_200_6h[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_200_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: uptrend (price > 200EMA on both timeframes) + RSI pullback from oversold + volume
            if (close[i] > ema_200_6h[i] and 
                close[i] > ema_200_12h_aligned[i] and 
                rsi[i] < 30 and 
                volume[i] > 1.2 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend (price < 200EMA on both timeframes) + RSI pullback from overbought + volume
            elif (close[i] < ema_200_6h[i] and 
                  close[i] < ema_200_12h_aligned[i] and 
                  rsi[i] > 70 and 
                  volume[i] > 1.2 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend break or RSI overextension
            if (close[i] < ema_200_6h[i] or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend break or RSI overextension
            if (close[i] > ema_200_6h[i] or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals