#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_volume_v2
# Strategy: Daily Camarilla pivot breakout with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Weekly trend filters reduce false breakouts on daily timeframe. Volume
# confirmation ensures breakouts have conviction. Targets 15-25 trades/year to avoid
# fee drag. Works in bull/bear by aligning with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous daily bar
    prev_close = prices['close'].shift(1).values
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    
    # Camarilla levels: H3, L3 (primary breakout levels)
    rng = prev_high - prev_low
    H3 = prev_close + 1.1 * rng / 4
    L3 = prev_close - 1.1 * rng / 4
    H4 = prev_close + 1.1 * rng / 2  # Stop/reverse levels
    L4 = prev_close - 1.1 * rng / 2
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 20-day volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(H4[i]) or np.isnan(L4[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout signals
        breakout_up = high[i] > H3[i-1]
        breakdown_down = low[i] < L3[i-1]
        
        # Weekly trend filter
        trend_bullish = close[i] > ema_50_1w_aligned[i]
        trend_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        # Long: Breakout above H3 AND bullish weekly trend AND volume confirmation
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below L3 AND bearish weekly trend AND volume confirmation
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout using H4/L4 levels
        elif position == 1 and low[i] < L4[i-1]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > H4[i-1]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals