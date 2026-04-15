#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and 1d volume confirmation
# Designed for low trade frequency (target 15-37/year) with clear mean-reversion logic
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
# Uses RSI(14) for entry timing, 4h EMA50 for trend direction, 1d volume spike for confirmation
# Includes session filter (08-20 UTC) to reduce noise trades
# Fixed position size of 0.20 to minimize fee churn

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate RSI(14) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA50 on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period average volume on 1d for volume confirmation
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1h timeframe
    rsi_aligned = rsi  # Already on 1h
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20  # Fixed position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            continue
        
        # Long entry: RSI oversold (<30) + uptrend (price > 4h EMA50) + volume spike
        if (rsi_aligned[i] < 30 and 
            close[i] > ema50_4h_aligned[i] and 
            volume[i] > 1.5 * vol_avg_1d_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: RSI overbought (>70) + downtrend (price < 4h EMA50) + volume spike
        elif (rsi_aligned[i] > 70 and 
              close[i] < ema50_4h_aligned[i] and 
              volume[i] > 1.5 * vol_avg_1d_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral zone (40-60) or opposite signal
        elif position == 1 and (rsi_aligned[i] > 40 or close[i] < ema50_4h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_aligned[i] < 60 or close[i] > ema50_4h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_RSI_4hEMA50_1dVolume_MeanReversion"
timeframe = "1h"
leverage = 1.0