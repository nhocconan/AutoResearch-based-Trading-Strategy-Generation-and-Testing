#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1d data for volume context
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # 1h ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h EMA20 for entry timing
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(ema20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 4h EMA34
        uptrend = close[i] > ema34_4h_aligned[i]
        downtrend = close[i] < ema34_4h_aligned[i]
        
        # Volume filter: current 1h volume > 1.3 * 20-day average volume
        volume_filter = volume[i] > (1.3 * volume_ma20_1d_aligned[i])
        
        # Entry timing: price crosses 1h EMA20 in direction of trend
        ema20_cross_up = close[i] > ema20[i] and close[i-1] <= ema20[i-1]
        ema20_cross_down = close[i] < ema20[i] and close[i-1] >= ema20[i-1]
        
        if position == 0:
            # Long entry: uptrend + volume + EMA20 cross up
            if uptrend and volume_filter and ema20_cross_up:
                signals[i] = 0.20
                position = 1
            # Short entry: downtrend + volume + EMA20 cross down
            elif downtrend and volume_filter and ema20_cross_down:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or volume drying up
            if not uptrend or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend reversal or volume drying up
            if not downtrend or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA34_Trend_Volume_EMA20Entry"
timeframe = "1h"
leverage = 1.0