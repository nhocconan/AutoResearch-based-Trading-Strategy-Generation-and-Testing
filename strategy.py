#!/usr/bin/env python3
# Hypothesis: 12h timeframe with daily KAMA trend filter and weekly volume confirmation.
# Uses KAMA (adaptive moving average) for trend detection that adapts to market conditions,
# weekly volume spike for institutional participation confirmation, and price position relative
# to KAMA for entry/exit. Designed to work in both bull and bear markets by following adaptive trend.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_KAMA_Trend_WeeklyVolume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - trend filter
    def calculate_kama(close_prices, length=10, fast=2, slow=30):
        # Efficiency ratio
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        # For simplicity, use rolling calculation
        change_series = pd.Series(change)
        volatility_series = pd.Series(volatility)
        er = change_series.rolling(window=length, min_periods=1).sum() / \
             volatility_series.rolling(window=length, min_periods=1).sum().replace(0, 1)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close_prices)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    # Calculate KAMA for trend
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    trend_up = close > kama
    trend_down = close < kama
    
    # Get weekly data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly average volume (20-week average)
    weekly_vol = df_1w['volume'].values
    avg_weekly_volume = pd.Series(weekly_vol).rolling(window=20, min_periods=20).mean().values
    # Align to 12h timeframe
    avg_weekly_volume_aligned = align_htf_to_ltf(prices, df_1w, avg_weekly_volume)
    # Current volume > 1.5x average weekly volume
    volume_confirmation = volume > (1.5 * avg_weekly_volume_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) + volume confirmation
            if trend_up[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + volume confirmation
            elif trend_down[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA (trend change)
            if not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA (trend change)
            if not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals