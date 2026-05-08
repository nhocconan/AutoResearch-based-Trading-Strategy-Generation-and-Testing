# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian breakout with daily trend filter and volume confirmation.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at opposite level).
# Uses weekly structure (less noise, fewer trades) with daily confirmation to avoid whipsaws.
# Target: 15-25 trades/year to minimize fee decay while capturing significant moves.

name = "1d_Weekly_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly high and low for Donchian(20)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels (20-week high/low)
    high_max_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    upper_band = align_htf_to_ltf(prices, df_1w, high_max_20w)
    lower_band = align_htf_to_ltf(prices, df_1w, low_min_20w)
    
    # Weekly trend filter: price above/below 20-week EMA
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up = (close_1w > ema20_1w).astype(float)
    trend_down = (close_1w < ema20_1w).astype(float)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down)
    
    # Daily volume spike: current volume > 1.5 * 20-day average
    vol_ma20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20d * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly upper band with volume spike and weekly uptrend
            long_cond = (close[i] > upper_band[i] and vol_spike[i] and trend_up_aligned[i] > 0.5)
            
            # Short entry: price breaks below weekly lower band with volume spike and weekly downtrend
            short_cond = (close[i] < lower_band[i] and vol_spike[i] and trend_down_aligned[i] > 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below weekly lower band (mean reversion to support)
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above weekly upper band (mean reversion to resistance)
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakout with daily trend filter and volume confirmation.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at opposite level).
# Uses weekly structure (less noise, fewer trades) with daily confirmation to avoid whipsaws.
# Target: 15-25 trades/year to minimize fee decay while capturing significant moves.