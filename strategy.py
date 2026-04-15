#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA200 mean reversion with 4h volume confirmation and 1d momentum filter
# Targets low trade frequency (15-30/year) by requiring confluence of multiple timeframes
# Works in bull markets (mean reversion from oversold) and bear markets (mean reversion from overbought)
# Uses volume spike to confirm institutional interest and daily momentum to avoid counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMA200 on 1h data
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Load 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    volume_4h = df_4h['volume'].values
    
    # Load 1d data for momentum filter (RSI)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 4h volume average (20-period)
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align indicators to 1h timeframe
    ema200_aligned = ema200  # already on 1h
    vol_avg_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_4h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Fixed position size to reduce churn
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is NaN
        if (np.isnan(vol_avg_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            continue
            
        # Session filter: only trade 8-20 UTC
        if not (8 <= hours[i] <= 20):
            continue
        
        # Long entry: price below EMA200 (oversold) + volume spike + bullish daily momentum
        if (close[i] < ema200[i] and 
            volume[i] > 2.0 * vol_avg_4h_aligned[i] and 
            rsi_1d_aligned[i] > 50 and  # Bullish momentum on daily
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price above EMA200 (overbought) + volume spike + bearish daily momentum
        elif (close[i] > ema200[i] and 
              volume[i] > 2.0 * vol_avg_4h_aligned[i] and 
              rsi_1d_aligned[i] < 50 and  # Bearish momentum on daily
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price crosses back above/below EMA200
        elif position == 1 and close[i] > ema200[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < ema200[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_EMA200_MeanReversion_4hVol_1dRSI"
timeframe = "1h"
leverage = 1.0