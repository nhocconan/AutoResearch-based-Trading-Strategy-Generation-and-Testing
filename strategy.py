#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with daily volume spike and weekly trend filter.
# Camarilla levels provide high-probability reversal/breakout zones.
# Daily volume spike confirms conviction at key levels.
# Weekly EMA50 filters for higher timeframe trend to avoid counter-trend trades.
# Target: 20-40 trades/year to minimize fee drag. Works in bull/bear via directional filter.

name = "4h_Camarilla_Pivot_Volume_WeeklyTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from previous daily bar
    # H, L, C from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.1 * (High - Low)
    # L3 = Close - 1.1 * (High - Low)
    range_hl = prev_high - prev_low
    h4 = prev_close + 1.5 * range_hl
    l4 = prev_close - 1.5 * range_hl
    h3 = prev_close + 1.1 * range_hl
    l3 = prev_close - 1.1 * range_hl
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Daily volume spike: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Weekly trend filter: EMA50 on weekly close
    weekly_close = df_1w['close'].values
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_weekly)
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above H3 with volume spike AND above weekly EMA50 (uptrend)
            long_breakout = close[i] > h3_aligned[i]
            uptrend = close[i] > ema50_aligned[i]
            if vol_spike and long_breakout and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume spike AND below weekly EMA50 (downtrend)
            elif vol_spike and close[i] < l3_aligned[i] and close[i] < ema50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 OR weekly trend turns down
            exit_condition = close[i] < l3_aligned[i] or close[i] < ema50_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H3 OR weekly trend turns up
            exit_condition = close[i] > h3_aligned[i] or close[i] > ema50_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals