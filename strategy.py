#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(2) extreme + 12h trend filter + volume confirmation
# Uses extremely short RSI to capture mean-reversion bounces in strong trends.
# Works in bull markets (long on RSI<10 in uptrend) and bear markets (short on RSI>90 in downtrend).
# Volume confirms momentum behind the move. Target: 80-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate RSI(2) on price
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA(50) on 12h for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period on 12h)
    vol_avg_12h = pd.Series(volume_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_avg_12h_aligned[i])):
            continue
        
        # Long entry: RSI < 10 (extremely oversold) + price above 12h EMA50 (uptrend) + volume spike
        if (rsi_aligned[i] < 10 and
            close[i] > ema50_12h_aligned[i] and
            volume[i] > 1.5 * vol_avg_12h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: RSI > 90 (extremely overbought) + price below 12h EMA50 (downtrend) + volume spike
        elif (rsi_aligned[i] > 90 and
              close[i] < ema50_12h_aligned[i] and
              volume[i] > 1.5 * vol_avg_12h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral (40-60) or opposite extreme
        elif position == 1 and (rsi_aligned[i] > 60 or rsi_aligned[i] > 90):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_aligned[i] < 40 or rsi_aligned[i] < 10):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_RSI2_Extreme_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0