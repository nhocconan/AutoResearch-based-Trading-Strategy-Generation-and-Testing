#!/usr/bin/env python3
name = "1d_SMA_Crossover_RSI_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # SMA fast and slow
    sma_fast = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    sma_slow = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    weekly_uptrend = close > sma_50_1w_aligned
    
    # Volume filter
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20, 14)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]) or np.isnan(rsi[i]) or np.isnan(weekly_uptrend[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: fast SMA > slow SMA, RSI > 50, weekly uptrend, volume
            if sma_fast[i] > sma_slow[i] and rsi[i] > 50 and weekly_uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: fast SMA < slow SMA, RSI < 50, weekly downtrend, volume
            elif sma_fast[i] < sma_slow[i] and rsi[i] < 50 and not weekly_uptrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: fast SMA < slow SMA or RSI < 40 or weekly downtrend
            if sma_fast[i] < sma_slow[i] or rsi[i] < 40 or not weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: fast SMA > slow SMA or RSI > 60 or weekly uptrend
            if sma_fast[i] > sma_slow[i] or rsi[i] > 60 or weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals