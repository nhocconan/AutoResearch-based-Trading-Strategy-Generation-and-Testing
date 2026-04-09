#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_breakout_volume_v1
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation (>2.0x 20-period average). Enters long when price breaks above 6h Donchian(20) high with volume confirmation and bullish weekly trend (price > weekly VWAP); short when price breaks below 6h Donchian(20) low with volume confirmation and bearish weekly trend (price < weekly VWAP). Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 12-37 trades/year) to work in both bull and bear markets by following institutional volume-driven breakouts in alignment with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly HTF data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation (using weekly data)
    prev_week_close = df_1w['close'].values
    prev_week_high = df_1w['high'].values
    prev_week_low = df_1w['low'].values
    prev_week_volume = df_1w['volume'].values
    
    # True range for the week
    week_range = prev_week_high - prev_week_low
    
    # Weekly pivot levels (standard calculation)
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - prev_week_low
    weekly_s1 = 2 * weekly_pivot - prev_week_high
    weekly_r2 = weekly_pivot + (prev_week_high - prev_week_low)
    weekly_s2 = weekly_pivot - (prev_week_high - prev_week_low)
    weekly_r3 = weekly_r2 + (prev_week_high - prev_week_low)
    weekly_s3 = weekly_s2 - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Weekly VWAP for trend filter (using typical price * volume)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    vwap_numerator = (typical_price * df_1w['volume']).cumsum().values
    vwap_denominator = df_1w['volume'].cumsum().values
    vwap = vwap_numerator / vwap_denominator
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    
    # 6h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_vwap_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches or breaks below weekly S1 level
            if close[i] <= weekly_s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks above weekly R1 level
            if close[i] >= weekly_r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and weekly trend alignment
            if volume_confirmed:
                # Bullish weekly trend: price above weekly VWAP
                bullish_trend = close[i] > weekly_vwap_aligned[i]
                # Bearish weekly trend: price below weekly VWAP
                bearish_trend = close[i] < weekly_vwap_aligned[i]
                
                # Long: price breaks above 6h Donchian high with volume and bullish weekly trend
                if close[i] > highest_high[i] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below 6h Donchian low with volume and bearish weekly trend
                elif close[i] < lowest_low[i] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals