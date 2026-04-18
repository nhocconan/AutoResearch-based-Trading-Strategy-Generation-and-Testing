#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Trend_Following
Hypothesis: Use weekly pivot levels to determine long-term trend direction, with 6h breakout entries.
In bull markets: trade long when price breaks above weekly R1 with volume confirmation.
In bear markets: trade short when price breaks below weekly S1 with volume confirmation.
Weekly pivot provides structural support/resistance that works across market regimes.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
"""

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
    
    # Get weekly data for pivot levels (primary HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for volume confirmation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly OHLC for pivot calculation (previous week)
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    # First week uses same week data
    prev_weekly_close[0] = weekly_close[0]
    prev_weekly_high[0] = weekly_high[0]
    prev_weekly_low[0] = weekly_low[0]
    
    # Weekly pivot levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    weekly_range = prev_weekly_high - prev_weekly_low
    weekly_r1 = prev_weekly_close + weekly_range * 1.1 / 12
    weekly_s1 = prev_weekly_close - weekly_range * 1.1 / 12
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    
    # Daily volume average for confirmation
    daily_volume = df_1d['volume'].values
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily close for trend filter (price above/below 20-day EMA)
    daily_close = df_1d['close'].values
    ema_20 = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly data to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Align daily data to 6h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(daily_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: price above/below daily 20 EMA
        price_above_ema = daily_close_aligned[i] > ema_20_aligned[i]
        price_below_ema = daily_close_aligned[i] < ema_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume and trend confirmation
            if (close[i] > weekly_r1_aligned[i] and vol_confirm and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume and trend confirmation
            elif (close[i] < weekly_s1_aligned[i] and vol_confirm and
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below weekly pivot or trend changes
            if (close[i] < weekly_pivot_aligned[i] or not price_above_ema):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly pivot or trend changes
            if (close[i] > weekly_pivot_aligned[i] or not price_below_ema):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Trend_Following"
timeframe = "6h"
leverage = 1.0