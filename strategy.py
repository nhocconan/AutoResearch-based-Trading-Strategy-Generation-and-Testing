#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
# Long when RSI < 30 (oversold), 4h close > 4h EMA50 (uptrend), and 1h volume > 2x 20-period average.
# Short when RSI > 70 (overbought), 4h close < 4h EMA50 (downtrend), and 1h volume > 2x 20-period average.
# Exit when RSI crosses back to 50 (mean reversion target).
# Uses RSI for mean reversion in 1h timeframe with trend filter to avoid counter-trend trades.
# Target: 100-150 total trades over 4 years (25-37/year) for 1h timeframe.
# Session filter: 08-20 UTC to reduce noise trades.

name = "1h_RSI_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # RSI (14-period) on 1h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for RSI and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema50_4h_aligned[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI < 30, 4h close > 4h EMA50, volume spike
            long_cond = (rsi[i] < 30) and (close_4h_aligned[i] > ema50_4h_aligned[i]) and volume_filter[i]
            # Short conditions: RSI > 70, 4h close < 4h EMA50, volume spike
            short_cond = (rsi[i] > 70) and (close_4h_aligned[i] < ema50_4h_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back above 50 (mean reversion target)
            if rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI crosses back below 50 (mean reversion target)
            if rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals