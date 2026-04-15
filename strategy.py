#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation
# Uses RSI extremes for mean reversion entries, filtered by 4h EMA trend direction,
# with volume confirmation to avoid false signals. Works in both bull and bear by
# only taking mean reversion trades in the direction of the 4h trend.
# Target: 80-150 total trades over 4 years (20-38/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA50 on 4h for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period on 1h)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h EMA50 to 1h
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(vol_avg[i])):
            continue
        
        # Long entry: RSI < 30 (oversold) + volume confirmation + price above 4h EMA50 (uptrend)
        if (rsi[i] < 30 and
            volume[i] > 1.2 * vol_avg[i] and
            close[i] > ema50_4h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: RSI > 70 (overbought) + volume confirmation + price below 4h EMA50 (downtrend)
        elif (rsi[i] > 70 and
              volume[i] > 1.2 * vol_avg[i] and
              close[i] < ema50_4h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral range (40-60) or opposite extreme
        elif position == 1 and (rsi[i] > 40):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 60):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0