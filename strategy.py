#!/usr/bin/env python3
# 1d_200EMA_5min_RSI_Pullback
# Hypothesis: On the daily chart, price above 200 EMA defines a bullish trend, below defines bearish.
# On the 5-minute chart, we look for pullbacks to the 20-period EMA with RSI < 30 (bull) or > 70 (bear)
# to enter in the direction of the daily trend. This combines trend following with mean reversion entries.
# Designed for low trade frequency (~10-20 trades/year) to minimize fee drag and work in both bull and bear markets.

name = "1d_200EMA_5min_RSI_Pullback"
timeframe = "5m"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for 200 EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 5m OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily 200 EMA for trend ---
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # --- 5m 20 EMA for pullback ---
    ema_20_5m = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # --- 5m RSI(14) for overbought/oversold ---
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for daily EMA200, 5m EMA20, and RSI
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(ema_20_5m[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend
        daily_uptrend = close[i] > ema_200_1d_aligned[i]
        daily_downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Pullback conditions
        near_ema = abs(close[i] - ema_20_5m[i]) / ema_20_5m[i] < 0.01  # Within 1% of 20 EMA
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # Long: daily uptrend + pullback to EMA20 with RSI oversold
            if daily_uptrend and near_ema and rsi_oversold:
                signals[i] = 0.25
                position = 1
            # Short: daily downtrend + pullback to EMA20 with RSI overbought
            elif daily_downtrend and near_ema and rsi_overbought:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below 20 EMA or RSI becomes overbought
                if close[i] < ema_20_5m[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above 20 EMA or RSI becomes oversold
                if close[i] > ema_20_5m[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals