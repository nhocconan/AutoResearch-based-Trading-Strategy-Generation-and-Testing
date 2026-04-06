#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h price action with 4h trend filter and volume confirmation
# Enter long when price crosses above 1h EMA(20) with volume > 1.3x average and price > 4h EMA(50)
# Enter short when price crosses below 1h EMA(20) with volume > 1.3x average and price < 4h EMA(50)
# Use 4h EMA(50) as trend filter to avoid counter-trend trades
# Exit when price crosses back over 1h EMA(20) or reverses against 4h trend
# Target: 60-150 trades over 4 years (15-37/year) with session filter (08-20 UTC)

name = "1h_ema20_4h_ema50_vol_session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h EMA(20) for entry signal
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    # Session filter: 08-20 UTC (pre-market to NY close)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(ema_20[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_threshold[i]) or not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below 1h EMA(20) OR closes below 4h EMA(50)
            if close[i] < ema_20[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price crosses above 1h EMA(20) OR closes above 4h EMA(50)
            if close[i] > ema_20[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: 1h EMA(20) cross with volume and trend filter
            if close[i] > ema_20[i] and volume[i] > volume_threshold[i] and close[i] > ema_50_4h_aligned[i]:
                # Long: price above EMA(20) with volume and uptrend
                signals[i] = 0.20
                position = 1
            elif close[i] < ema_20[i] and volume[i] > volume_threshold[i] and close[i] < ema_50_4h_aligned[i]:
                # Short: price below EMA(20) with volume and downtrend
                signals[i] = -0.20
                position = -1
    
    return signals