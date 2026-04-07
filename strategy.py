#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation
# Uses Donchian breakouts for trend capture, 12h EMA(50) for trend direction filter,
# and volume > 1.5x 20-period average for confirmation. Designed for low trade frequency
# (target: 15-35 trades/year) to minimize fee drift. Works in bull markets via breakout
# continuation and in bear markets via breakdown continuation with trend filter.

name = "6h_donchian20_12h_trend_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # break above prior 20-period high
        breakdown_down = close[i] < lowest_low[i-1]  # break below prior 20-period low
        
        # Trend filter from 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Long: upward breakout + uptrend + volume confirmation
        if breakout_up and uptrend and vol_confirm:
            signals[i] = 0.25
        # Short: downward breakdown + downtrend + volume confirmation
        elif breakdown_down and downtrend and vol_confirm:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals