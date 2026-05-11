#!/usr/bin/env python3
name = "6h_RSI_MeanReversion_1wTrend_Volume"
timeframe = "6h"
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
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up_1w = close_1w > ema200_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # RSI(14) for mean reversion
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for RSI and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(rsi[i]) or np.isnan(trend_up_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + weekly uptrend + volume confirmation
            if rsi[i] < 30 and trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + weekly downtrend + volume confirmation
            elif rsi[i] > 70 and not trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought (>70) OR weekly trend turns down
            if rsi[i] > 70 or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold (<30) OR weekly trend turns up
            if rsi[i] < 30 or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals