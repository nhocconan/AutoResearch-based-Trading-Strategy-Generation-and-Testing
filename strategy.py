#!/usr/bin/env python3
# Hypothesis: Daily RSI mean reversion with weekly trend filter and volume confirmation.
# In strong weekly trends (price above/below 50-week EMA), daily RSI extremes (>70/<30) often reverse.
# Enters short when weekly uptrend + daily RSI > 70, long when weekly downtrend + daily RSI < 30.
# Uses volume confirmation: only enter if current volume > 1.5x 20-day average volume.
# Exits when RSI returns to neutral (40-60 range) or weekly trend changes.
# Target: 20-60 total trades over 4 years (5-15/year) with size 0.25.

name = "1d_RSI_MeanReversion_WeeklyTrend"
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
    volume = prices['volume'].values
    
    # Weekly trend: 50-period EMA on weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly trend direction: price above/below EMA50
    weekly_uptrend = close > ema_50_1w_aligned
    weekly_downtrend = close < ema_50_1w_aligned
    
    # Daily RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter short: weekly uptrend + RSI > 70 + volume confirmation
            if weekly_uptrend[i] and (rsi[i] > 70) and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            # Enter long: weekly downtrend + RSI < 30 + volume confirmation
            elif weekly_downtrend[i] and (rsi[i] < 30) and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or weekly trend changes to uptrend
            if (rsi[i] >= 40) or weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or weekly trend changes to downtrend
            if (rsi[i] <= 60) or weekly_downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals