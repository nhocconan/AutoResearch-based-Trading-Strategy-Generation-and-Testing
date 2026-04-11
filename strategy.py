#!/usr/bin/env python3
# 12h_1d_donchian_breakout_v1
# Strategy: 12h Donchian breakout with 1d EMA trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture breakout moves. 1d EMA200 filters trend direction.
# Volume confirmation ensures institutional participation. Designed for low trade frequency
# (~15-30/year) to minimize fee drift. Works in bull markets via long breakouts above EMA200
# and bear markets via short breakdowns below EMA200.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Wait for EMA200 warmup
        # Skip if any required data is invalid
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or \
           np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period high
        breakdown_down = close[i] < donchian_low[i-1]  # Break below previous period low
        
        # Trend filter: price above/below 1d EMA200
        price_above_ema = close[i] > ema_200_1d_aligned[i]
        price_below_ema = close[i] < ema_200_1d_aligned[i]
        
        # Entry conditions
        # Long: bullish breakout AND price above EMA200 AND volume confirmation
        if breakout_up and price_above_ema and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: bearish breakdown AND price below EMA200 AND volume confirmation
        elif breakdown_down and price_below_ema and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout (trend reversal)
        elif position == 1 and breakdown_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals