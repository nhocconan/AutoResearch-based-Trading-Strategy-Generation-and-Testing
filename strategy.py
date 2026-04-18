#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation
# Donchian breakouts capture momentum moves with clear entry/exit rules.
# 4h trend filter (EMA50) ensures trades align with higher timeframe trend.
# Volume confirmation filters out false breakouts.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
name = "1h_Donchian_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Donchian channels (20-period) on 1h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume filter: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_max = high_max_20[i]
        low_min = low_min_20[i]
        ema_trend = ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: Breakout above upper Donchian band AND above 4h EMA50 AND volume filter
            if close_val > high_max and close_val > ema_trend and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: Breakdown below lower Donchian band AND below 4h EMA50 AND volume filter
            elif close_val < low_min and close_val < ema_trend and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Close below lower Donchian band (breakdown) or above 4h EMA50 (trailing)
            if close_val < low_min or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Close above upper Donchian band (breakout) or below 4h EMA50 (trailing)
            if close_val > high_max or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals