#!/usr/bin/env python3
"""
6h_WilliamsVixFix_Breakout_WeeklyTrend
Hypothesis: Williams Vix Fix (WVF) identifies volatility expansion and potential reversal points on 6h timeframe. Combined with weekly trend filter (price above/below weekly EMA20) and volume confirmation, this strategy captures high-probability breakouts in both bull and bear markets. WVF > 0.8 signals extreme fear/greed and impending reversals. Weekly trend ensures directional alignment with higher timeframe momentum. Volume spike (>2.0x 20-bar average) confirms institutional participation. Discrete sizing (0.25) limits fee churn. Target: 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Williams Vix Fix calculation (using 22-period lookback as per original)
    # WVF = ((Highest Close in last 22 periods - Low) / (Highest Close in last 22 periods - Lowest Close in last 22 periods)) * 100
    lookback = 22
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    denominator = highest_close - lowest_close
    wvf = np.where(denominator != 0, ((highest_close - low) / denominator) * 100, 0)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(wvf[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_trend = ema_20_1w_aligned[i]
        wvf_val = wvf[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        low_val = low[i]
        high_val = high[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        # WVF extreme condition: > 0.8 (80%) indicates high volatility/reversal potential
        wvf_extreme = wvf_val > 80.0
        
        if position == 0:
            # Look for entry signals: WVF extreme + volume spike + trend alignment
            # Long: WVF extreme (fear spike) + price above weekly EMA + volume spike
            long_signal = wvf_extreme and (close_val > ema_trend) and volume_spike
            # Short: WVF extreme (greed spike) + price below weekly EMA + volume spike
            short_signal = wvf_extreme and (close_val < ema_trend) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. WVF normalizes (< 50) indicating reduced fear
            # 2. Price crosses below weekly EMA (trend change)
            if (wvf_val < 50.0) or (close_val < ema_trend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. WVF normalizes (< 50) indicating reduced greed
            # 2. Price crosses above weekly EMA (trend change)
            if (wvf_val < 50.0) or (close_val > ema_trend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVixFix_Breakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0