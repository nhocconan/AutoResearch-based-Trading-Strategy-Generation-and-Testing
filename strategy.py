#!/usr/bin/env python3
name = "12h_1d_RSI_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(14) on 12h close
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
    
    # 1-day SMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Bollinger Bands(20,2) on 12h close for mean reversion zones
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    lower_band = sma_20 - 2 * std_20
    upper_band = sma_20 + 2 * std_20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(14, 20, 50)
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(sma_50_1d_aligned[i]) or np.isnan(lower_band[i]) or np.isnan(upper_band[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + price below lower BB + above 1d SMA50 (bullish bias)
            if rsi[i] < 30 and close[i] < lower_band[i] and close[i] > sma_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + price above upper BB + below 1d SMA50 (bearish bias)
            elif rsi[i] > 70 and close[i] > upper_band[i] and close[i] < sma_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 or price back above middle band
            if rsi[i] > 50 or close[i] > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 50 or price back below middle band
            if rsi[i] < 50 or close[i] < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals