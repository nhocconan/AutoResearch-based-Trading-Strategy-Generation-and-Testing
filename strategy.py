#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_Filter_v2"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate weekly trend filter (100 EMA)
    df_1w = get_htf_data(prices, '1w')
    ema100_1w = pd.Series(df_1w['close']).ewm(span=100, min_periods=100, adjust=False).mean().values
    ema100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    # Calculate weekly pivot points (daily high/low/close aggregated to week)
    # For weekly pivot, we need weekly OHLC
    df_1w_ohlc = get_htf_data(prices, '1w')
    weekly_high = df_1w_ohlc['high'].values
    weekly_low = df_1w_ohlc['low'].values
    weekly_close = df_1w_ohlc['close'].values
    
    # Weekly pivot calculation
    pivot_weekly = (weekly_high + weekly_low + weekly_close) / 3
    range_weekly = weekly_high - weekly_low
    R1_weekly = pivot_weekly + (range_weekly * 1.1 / 12)
    S1_weekly = pivot_weekly - (range_weekly * 1.1 / 12)
    R2_weekly = pivot_weekly + (range_weekly * 1.1 / 6)
    S2_weekly = pivot_weekly - (range_weekly * 1.1 / 6)
    
    # Align weekly data to 6h timeframe
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_1w_ohlc, pivot_weekly)
    R1_weekly_aligned = align_htf_to_ltf(prices, df_1w_ohlc, R1_weekly)
    S1_weekly_aligned = align_htf_to_ltf(prices, df_1w_ohlc, S1_weekly)
    R2_weekly_aligned = align_htf_to_ltf(prices, df_1w_ohlc, R2_weekly)
    S2_weekly_aligned = align_htf_to_ltf(prices, df_1w_ohlc, S2_weekly)
    
    # Volume filter: 24-period EMA (4 days worth)
    vol_ema24 = pd.Series(volume).ewm(span=24, min_periods=24, adjust=False).mean().values
    volume_ok = volume > vol_ema24 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema100_1w_aligned[i]) or np.isnan(pivot_weekly_aligned[i]) or 
            np.isnan(R1_weekly_aligned[i]) or np.isnan(S1_weekly_aligned[i]) or
            np.isnan(R2_weekly_aligned[i]) or np.isnan(S2_weekly_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_weekly_ema = close[i] > ema100_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema100_1w_aligned[i]
        breakout_long = close[i] > R2_weekly_aligned[i]
        breakout_short = close[i] < S2_weekly_aligned[i]
        pullback_long = close[i] > R1_weekly_aligned[i] and close[i] < R2_weekly_aligned[i]
        pullback_short = close[i] < S1_weekly_aligned[i] and close[i] > S2_weekly_aligned[i]
        
        if position == 0:
            # Long: Break above R2 with weekly uptrend + volume
            if breakout_long and price_above_weekly_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S2 with weekly downtrend + volume
            elif breakout_short and price_below_weekly_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            # Long pullback: Pullback to R1 in uptrend
            elif pullback_long and price_above_weekly_ema and volume_ok[i]:
                signals[i] = 0.20
                position = 1
            # Short pullback: Pullback to S1 in downtrend
            elif pullback_short and price_below_weekly_ema and volume_ok[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price breaks below S1 OR trend reverses
                if close[i] < S1_weekly_aligned[i] or close[i] < ema100_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: Price breaks above R1 OR trend reverses
                if close[i] > R1_weekly_aligned[i] or close[i] > ema100_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals