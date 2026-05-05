#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI with 12h Trend Filter
# Long when: VW-RSI(14) < 30 AND 12h EMA(50) > 12h EMA(200) (bullish trend)
# Short when: VW-RSI(14) > 70 AND 12h EMA(50) < 12h EMA(200) (bearish trend)
# Exit when: VW-RSI crosses 50 (mean reversion to midpoint)
# VW-RSI reduces noise by weighting price changes with volume, improving signal quality
# 12h EMA cross provides multi-timeframe trend alignment to avoid counter-trend whipsaws
# Works in both bull and bear markets by only trading oversold/overbott in direction of higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_VolumeWeightedRSI_12hEMATrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:  # Need enough for EMA(200)
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMAs for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate Volume-Weighted RSI (14) on 6h
    # Weight price changes by volume to reduce noise
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta * volume, 0)
    loss = np.where(delta < 0, -delta * volume, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(vw_rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h EMA(50) > EMA(200) = bullish, < = bearish
        bullish_trend = ema_50_12h_aligned[i] > ema_200_12h_aligned[i]
        bearish_trend = ema_50_12h_aligned[i] < ema_200_12h_aligned[i]
        
        if position == 0:
            # Long: Oversold VW-RSI in bullish 12h trend
            if vw_rsi[i] < 30 and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: Overbought VW-RSI in bearish 12h trend
            elif vw_rsi[i] > 70 and bearish_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VW-RSI crosses above 50 (mean reversion)
            if vw_rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VW-RSI crosses below 50 (mean reversion)
            if vw_rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals