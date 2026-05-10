#!/usr/bin/env python3
# 4H_Relative_Strength_Index_RSI_Overbought_Oversold_1dTrend_Volume
# Hypothesis: Uses daily trend filter with RSI(14) extremes for mean reversion entries. 
# In bull markets (price > 1d EMA50), look for RSI < 30 (oversold) long entries.
# In bear markets (price < 1d EMA50), look for RSI > 70 (overbought) short entries.
# Volume confirmation (>2x average) filters low-quality signals. Designed for 4h timeframe
# to capture mean reversion moves within the dominant daily trend, reducing counter-trend trades.
# Works in both bull and bear markets by aligning with 1d trend direction. Target: 25-40 trades/year.

name = "4H_RSI_Overbought_Oversold_1dTrend_Volume"
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
    
    # Get 1d data for EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # fillna for stability
    
    # Volume filter: volume > 2x 20-period average on 4h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # Long entry: price above 1d EMA (bull trend) + RSI < 30 (oversold) + volume spike
            if (price_above_ema and 
                rsi_values[i] < 30 and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below 1d EMA (bear trend) + RSI > 70 (overbought) + volume spike
            elif (price_below_ema and 
                  rsi_values[i] > 70 and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) or volume drops below average
            if (rsi_values[i] > 50 or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) or volume drops below average
            if (rsi_values[i] < 50 or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals