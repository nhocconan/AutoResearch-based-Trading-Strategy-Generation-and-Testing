#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d Trend Reversal with Volume Spike
# Uses 4h EMA for trend direction and 1d RSI for overbought/oversold conditions.
# Enters on 1h pullbacks to 4h EMA with volume spike confirmation during extreme RSI readings.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 80-150 total trades over 4 years = 20-38/year for 1h.
# Timeframe: 1h, HTF: 4h/1d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate session filter (08-20 UTC) once
    hours = prices.index.hour
    
    # Load 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data for RSI overbought/oversold
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # Calculate RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            continue
            
        # Apply session filter (08-20 UTC)
        hour = hours[i]
        if hour < 8 or hour > 20:
            continue
        
        # Long entry: price near 4h EMA during oversold RSI with volume spike
        if (close[i] >= ema_4h_aligned[i] * 0.995 and close[i] <= ema_4h_aligned[i] * 1.005 and
            rsi_1d_aligned[i] < 30 and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price near 4h EMA during overbought RSI with volume spike
        elif (close[i] >= ema_4h_aligned[i] * 0.995 and close[i] <= ema_4h_aligned[i] * 1.005 and
              rsi_1d_aligned[i] > 70 and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or RSI returns to neutral
        elif position == 1 and (rsi_1d_aligned[i] > 50 or close[i] < ema_4h_aligned[i] * 0.98):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_1d_aligned[i] < 50 or close[i] > ema_4h_aligned[i] * 1.02):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_4h1d_TrendReversal_VolumeSpike"
timeframe = "1h"
leverage = 1.0