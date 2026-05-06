# 6h_RSI45_55_Band_12hTrend_Filter_v1
# RSI band (45-55) on 6h with 12h EMA200 trend filter and volume confirmation
# Captures mean reversion in strong trends: long when RSI oversold in uptrend, short when RSI overbought in downtrend
# Volume filter ensures participation, trend filter reduces whipsaw
# Works in bull/bear: follows major trend while fading extremes
# Target: 50-150 total trades over 4 years (12-37/year)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI45_55_Band_12hTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA200 trend filter
    close_12h_series = pd.Series(close_12h)
    ema200_12h = close_12h_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate RSI(14) on 6h
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[50], rsi])  # align length
    
    # Calculate volume filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe (primary)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema200_12h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 45 (oversold) AND uptrend (price > EMA200) AND volume
            if rsi[i] < 45 and close[i] > ema200_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 55 (overbought) AND downtrend (price < EMA200) AND volume
            elif rsi[i] > 55 and close[i] < ema200_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 55 (overbought) or trend change
            if rsi[i] > 55 or close[i] <= ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 45 (oversold) or trend change
            if rsi[i] < 45 or close[i] >= ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals