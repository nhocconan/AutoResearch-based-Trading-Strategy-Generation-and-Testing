#!/usr/bin/env python3
# 1D_RSI_Trend_Reversal_4H_Trend
# Hypothesis: Intraday RSI mean reversion on 1d timeframe, filtered by 4h trend direction to avoid counter-trend trades in both bull and bear markets.
# Uses RSI(14) < 30 for long, RSI(14) > 70 for short, only in direction of 4h EMA(50) trend.
# Low trade frequency expected (<25/year) due to strict RSI extremes + trend filter.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "1D_RSI_Trend_Reversal_4H_Trend"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1d timeframe
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # RSI warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 4h EMA50
        price_above_ema = close[i] > ema_4h_aligned[i]
        price_below_ema = close[i] < ema_4h_aligned[i]
        
        if position == 0:
            # Long entry: RSI < 30 (oversold) + price above 4h EMA (uptrend)
            if rsi_values[i] < 30 and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI > 70 (overbought) + price below 4h EMA (downtrend)
            elif rsi_values[i] > 70 and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion) or trend change
            if rsi_values[i] > 50 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion) or trend change
            if rsi_values[i] < 50 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals