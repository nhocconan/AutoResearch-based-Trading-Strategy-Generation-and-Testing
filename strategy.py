#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_donchian_breakout_volume_trend_v1
# 4-hour Donchian breakout with 12-hour trend confirmation (EMA50) and volume filter.
# In bull markets: breakouts above upper band with volume and uptrend signal long.
# In bear markets: breakdowns below lower band with volume and downtrend signal short.
# Uses volume confirmation to avoid false breakouts and trend filter to align with higher timeframe momentum.
# Target: 20-40 trades/year per symbol for low friction and high edge.
name = "4h_12h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if Donchian levels not ready
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
        
        # Skip if volume or trend filter fails
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend from 12h EMA50
        if np.isnan(ema_50_12h_aligned[i]):
            trend_up = False
            trend_down = False
        else:
            trend_up = close[i] > ema_50_12h_aligned[i]
            trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Long signal: breakout above upper Donchian with volume and uptrend
        if high[i] > highest_high[i] and vol_confirm[i] and trend_up and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: breakdown below lower Donchian with volume and downtrend
        elif low[i] < lowest_low[i] and vol_confirm[i] and trend_down and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif low[i] < lowest_low[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif high[i] > highest_high[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals