#!/usr/bin/env python3
"""
1D_RSI_Extreme_1wTrend_Volume
Hypothesis: Uses weekly trend direction with daily RSI extremes for mean reversion entries, filtered by volume spikes. Works in both bull and bear markets by following weekly trend direction for entries but using daily RSI extremes for counter-trend entries within the trend. Targets 15-30 trades/year with discrete sizing to minimize fee drag.
"""

name = "1D_RSI_Extreme_1wTrend_Volume"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend direction
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA20
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long entry: RSI oversold (<30) + above weekly EMA + volume spike
            if (rsi_values[i] < 30 and 
                price_above_weekly_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (>70) + below weekly EMA + volume spike
            elif (rsi_values[i] > 70 and 
                  price_below_weekly_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought (>70) or price below weekly EMA
            if (rsi_values[i] > 70 or 
                close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold (<30) or price above weekly EMA
            if (rsi_values[i] < 30 or 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals