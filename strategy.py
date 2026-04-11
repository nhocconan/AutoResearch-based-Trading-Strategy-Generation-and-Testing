#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v1
# Strategy: Daily timeframe using weekly Camarilla pivot breakouts with volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Weekly Camarilla levels provide strong support/resistance. Daily breakouts
# with volume confirmation capture multi-day moves. Designed for low trade frequency
# (7-25/year) to minimize fee drag and work in both bull and bear markets via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous weekly bar
    # Using close of previous week as base
    prev_close = df_1w['close'].shift(1).values  # Previous week's close
    prev_high = df_1w['high'].shift(1).values    # Previous week's high
    prev_low = df_1w['low'].shift(1).values      # Previous week's low
    
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1
    # Range = prev_high - prev_low
    # H4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    # L4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    # H3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    # L3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    # H2 = prev_close + 1.1 * (prev_high - prev_low) / 6
    # L2 = prev_close - 1.1 * (prev_high - prev_low) / 6
    # H1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    # L1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    rng = prev_high - prev_low
    H4 = prev_close + 1.1 * rng / 2
    L4 = prev_close - 1.1 * rng / 2
    H3 = prev_close + 1.1 * rng / 4
    L3 = prev_close - 1.1 * rng / 4
    H2 = prev_close + 1.1 * rng / 6
    L2 = prev_close - 1.1 * rng / 6
    H1 = prev_close + 1.1 * rng / 12
    L1 = prev_close - 1.1 * rng / 12
    
    # Align Camarilla levels to daily timeframe (use previous week's levels)
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    
    # Weekly EMA20 for trend filter (using weekly close)
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 20-period volume average for confirmation (daily)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume[i] > 1.8 * vol_avg_20[i]
        
        # Breakout signals using weekly Camarilla levels
        # Breakout above H3 (strong resistance)
        breakout_up = high[i] > H3_aligned[i-1]
        # Breakdown below L3 (strong support)
        breakdown_down = low[i] < L3_aligned[i-1]
        
        # Weekly EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_20_1w_aligned[i]
        trend_bearish = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        # Long: Breakout above H3 AND bullish trend AND volume confirmation
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below L3 AND bearish trend AND volume confirmation
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout using H4/L4 levels (stronger levels)
        elif position == 1 and low[i] < L4_aligned[i-1]:  # Break below L4
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > H4_aligned[i-1]:  # Break above H4
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals