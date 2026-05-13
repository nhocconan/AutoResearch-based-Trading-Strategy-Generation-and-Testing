#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-day high AND close > 1w EMA50 AND volume > 1.5x 20-day average volume.
# Short when price breaks below 20-day low AND close < 1w EMA50 AND volume > 1.5x 20-day average volume.
# Exit when price crosses the 10-day EMA in the opposite direction.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness in trending markets.
# Primary timeframe: 1d, HTF: 1w. Target trades: 20-50 over 4 years (5-12/year) to avoid fee drag.

name = "1d_Donchian20_EMA50_Volume_Trend_v1"
timeframe = "1d"
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
    
    # Calculate 1w EMA50 for trend filter (HTF) - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-day Donchian channels (primary TF)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Calculate 20-day average volume for volume confirmation
    avg_vol = np.full(n, np.nan)
    for i in range(lookback, n):
        avg_vol[i] = np.mean(volume[i-lookback:i])
    
    # Calculate 10-day EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(avg_vol[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema_10[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above 20-day high AND price > 1w EMA50 AND volume > 1.5x average volume
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * avg_vol[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below 20-day low AND price < 1w EMA50 AND volume > 1.5x average volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_vol[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 10-day EMA
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 10-day EMA
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals