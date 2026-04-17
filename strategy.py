#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Weekly Pivot + 1d EMA200 Filter with Volume Spike
# Uses weekly pivot points (P, R1, S1) and daily EMA200 to filter trend direction.
# Enters long when price breaks above weekly R1 with volume spike and above daily EMA200.
# Enters short when price breaks below weekly S1 with volume spike and below daily EMA200.
# Weekly pivot provides structural levels, daily EMA200 filters trend, volume spike confirms.
# Designed for low turnover (target: 15-30 trades/year) to minimize fee drag on 6h timeframe.
# Works in bull markets (breakout momentum) and bear markets (mean reversion via pivot rejection).

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Get daily data for EMA200
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate weekly pivot points (P, R1, S1) from previous week
    # Pivot = (H + L + C) / 3
    # R1 = (2 * P) - L
    # S1 = (2 * P) - H
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3
    weekly_r1 = (2 * weekly_pivot) - low_weekly
    weekly_s1 = (2 * weekly_pivot) - high_weekly
    
    # Calculate daily EMA200 for trend filter
    close_daily_series = pd.Series(close_daily)
    ema200_daily = close_daily_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly and daily indicators to 6h timeframe
    weekly_r1_6h = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_pivot_6h = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    ema200_6h = align_htf_to_ltf(prices, df_daily, ema200_daily)
    
    # Volume filter: current volume > 2.0 * 20-period average (strict to reduce trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need sufficient data for EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or 
            np.isnan(ema200_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below daily EMA200
        price_above_ema = close[i] > ema200_6h[i]
        price_below_ema = close[i] < ema200_6h[i]
        
        # Price relative to weekly pivot levels
        price_above_r1 = close[i] > weekly_r1_6h[i]
        price_below_s1 = close[i] < weekly_s1_6h[i]
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume and above daily EMA200
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume and below daily EMA200
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly pivot OR below daily EMA200
            if (close[i] < weekly_pivot_6h[i]) or (close[i] < ema200_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly pivot OR above daily EMA200
            if (close[i] > weekly_pivot_6h[i]) or (close[i] > ema200_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6s_WeeklyPivot_1dEMA200_Volume"
timeframe = "6h"
leverage = 1.0