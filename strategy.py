#!/usr/bin/env python3
# 4h_12h_camarilla_breakout_v1
# Strategy: 4h Camarilla pivot breakout with 12h trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels (H4/L4) act as key support/resistance on 12h. 
# Breakout above H4 with 12h EMA20 > EMA50 and volume > 1.5x 20-period average triggers long.
# Breakdown below L4 with 12h EMA20 < EMA50 and volume confirmation triggers short.
# Uses tight entry conditions to limit trades (~20-40/year) and avoid fee drag.
# Works in bull markets via breakout continuation and bear markets via breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA20 and EMA50 for trend filter
    ema_20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_12h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d typical price for Camarilla calculation
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    tp_high = typical_price_1d.max()
    tp_low = typical_price_1d.min()
    tp_close = typical_price_1d.iloc[-1]
    
    # Calculate Camarilla levels for 1d
    range_1d = tp_high - tp_low
    h4 = tp_close + range_1d * 1.1 / 2
    l4 = tp_close - range_1d * 1.1 / 2
    
    # Broadcast Camarilla levels to all 1d bars (they're constant for the day)
    h4_1d = np.full_like(df_1d['close'].values, h4)
    l4_1d = np.full_like(df_1d['close'].values, l4)
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # 12h volume average (20-period) for confirmation
    volume_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Align raw 12h volume for confirmation
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_20_12h[i]) or np.isnan(ema_50_12h[i]) or \
           np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or \
           np.isnan(vol_avg_20_12h_aligned[i]) or np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        vol_confirm = vol_12h_aligned[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Trend filter: 12h EMA20 > EMA50 for bullish, < for bearish
        trend_bullish = ema_20_12h[i] > ema_50_12h[i]
        trend_bearish = ema_20_12h[i] < ema_50_12h[i]
        
        # Breakout conditions
        breakout_long = close[i] > h4_aligned[i] and trend_bullish and vol_confirm
        breakdown_short = close[i] < l4_aligned[i] and trend_bearish and vol_confirm
        
        # Entry conditions
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakdown_short and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout/breakdown
        elif position == 1 and close[i] < l4_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > h4_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals