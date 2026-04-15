#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Mean Reversion with 4h Trend Filter and Volume Confirmation
# Uses RSI(2) for extreme mean reversion signals on 1h, filtered by 4h EMA trend direction.
# Only takes long when price > 4h EMA50 and RSI(2) < 10, short when price < 4h EMA50 and RSI(2) > 90.
# Volume confirmation requires current volume > 1.5x 20-bar median volume.
# Works in both bull and bear markets by following the 4h trend direction.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h close
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(2) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(2, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]):
            continue
        
        # Volume confirmation: current volume > 1.5x 20-bar median
        vol_median = np.median(volume[max(0, i-19):i+1])
        vol_confirmed = volume[i] > 1.5 * vol_median
        
        # Long entry: price > 4h EMA50 (uptrend) + RSI(2) < 10 (oversold) + volume
        if (close[i] > ema_50_4h_aligned[i] and
            rsi[i] < 10 and
            vol_confirmed and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price < 4h EMA50 (downtrend) + RSI(2) > 90 (overbought) + volume
        elif (close[i] < ema_50_4h_aligned[i] and
              rsi[i] > 90 and
              vol_confirmed and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral zone (40-60) or trend reversal
        elif position == 1 and (rsi[i] > 40 or close[i] < ema_50_4h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 60 or close[i] > ema_50_4h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_RSI2_4hEMA50_Volume_MeanReversion"
timeframe = "1h"
leverage = 1.0