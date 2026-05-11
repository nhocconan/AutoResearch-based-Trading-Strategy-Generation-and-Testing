#!/usr/bin/env python3
# 4h_1d_200EMA_RSI_Pullback
# Hypothesis: Buy pullbacks to the 200 EMA in uptrends and sell rallies to the 200 EMA in downtrends,
# using RSI to avoid catching falling knives. Uses 1d EMA200 for trend and 4h RSI(14) for entry timing.
# Designed for low trade frequency (15-30/year) to minimize fee drag and work in both bull and bear markets.

name = "4h_1d_200EMA_RSI_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data for 200 EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d EMA200 for trend
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Shift RSI by 1 to avoid look-ahead
    rsi_prev = np.roll(rsi, 1)
    rsi_prev[0] = 50  # neutral start
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(rsi_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA200
        price_above_ema200 = close[i] > ema_200_1d_aligned[i]
        price_below_ema200 = close[i] < ema_200_1d_aligned[i]
        
        if position == 0:
            # Long: pullback to EMA200 in uptrend when RSI is oversold
            if price_above_ema200 and rsi_prev[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: rally to EMA200 in downtrend when RSI is overbought
            elif price_below_ema200 and rsi_prev[i] > 70:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long when price crosses above EMA200 (trend change) or RSI overbought
                if close[i] < ema_200_1d_aligned[i] or rsi_prev[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when price crosses below EMA200 (trend change) or RSI oversold
                if close[i] > ema_200_1d_aligned[i] or rsi_prev[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals