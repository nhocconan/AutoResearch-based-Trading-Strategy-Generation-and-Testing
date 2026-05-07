#!/usr/bin/env python3
# 6h_WeeklyPivot_RangeReversal_Terminator
# Hypothesis: Weekly pivot-based range reversal strategy for 6h timeframe. Uses weekly pivot points (PP) and support/resistance levels (S1, R1, S2, R2) calculated from prior week's OHLC. Enters long when price touches S1/S2 with bullish rejection (close > open) and weekly trend filter (price > weekly EMA20). Enters short when price touches R1/R2 with bearish rejection (close < open) and weekly trend filter (price < weekly EMA20). Weekly trend filter avoids counter-trend trades in strong trends. Uses volume confirmation (volume > 1.5x 20-period average) to avoid low-conviction moves. Designed to work in ranging markets (mean reversion at weekly S/R) and trending markets (pullbacks to weekly S/R in direction of weekly trend). Targets 15-25 trades/year per symbol to minimize fee drag.

name = "6h_WeeklyPivot_RangeReversal_Terminator"
timeframe = "6h"
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:  # Need enough data for EMA20
        return np.zeros(n)
    
    # Calculate weekly pivot points (based on previous week's OHLC)
    prev_week_high = df_w['high'].shift(1).values
    prev_week_low = df_w['low'].shift(1).values
    prev_week_close = df_w['close'].shift(1).values
    prev_week_open = df_w['open'].shift(1).values  # Not used in calc but kept for consistency
    
    # Weekly pivot point and support/resistance levels
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pp - prev_week_low
    s1 = 2 * pp - prev_week_high
    r2 = pp + (prev_week_high - prev_week_low)
    s2 = pp - (prev_week_high - prev_week_low)
    
    # Calculate weekly EMA20 for trend filter
    ema_20 = pd.Series(prev_week_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    ema_20_aligned = align_htf_to_ltf(prices, df_w, ema_20)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure we have EMA20 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: above average volume
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price at S1 or S2 with bullish rejection and uptrend
            at_s1 = abs(low[i] - s1_aligned[i]) < (pp_aligned[i] - s1_aligned[i]) * 0.05  # Within 5% of S1
            at_s2 = abs(low[i] - s2_aligned[i]) < (pp_aligned[i] - s2_aligned[i]) * 0.05  # Within 5% of S2
            bullish_rejection = close[i] > open_price[i]  # Bullish candle
            uptrend = close[i] > ema_20_aligned[i]
            
            if ((at_s1 or at_s2) and bullish_rejection and uptrend and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price at R1 or R2 with bearish rejection and downtrend
            elif (abs(high[i] - r1_aligned[i]) < (r1_aligned[i] - pp_aligned[i]) * 0.05 or  # Within 5% of R1
                  abs(high[i] - r2_aligned[i]) < (r2_aligned[i] - pp_aligned[i]) * 0.05):  # Within 5% of R2
                bearish_rejection = close[i] < open_price[i]  # Bearish candle
                downtrend = close[i] < ema_20_aligned[i]
                if bearish_rejection and downtrend and volume_filter:
                    signals[i] = -0.25
                    position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Price reaches opposite support/resistance level (take profit)
            # 2. Price crosses weekly pivot point (mean reversion completion)
            # 3. Trend filter fails (price crosses weekly EMA20 against position)
            tp_long = (position == 1 and (high[i] >= r1_aligned[i] or high[i] >= r2_aligned[i]))
            tp_short = (position == -1 and (low[i] <= s1_aligned[i] or low[i] <= s2_aligned[i]))
            at_pivot = abs(close[i] - pp_aligned[i]) < (r1_aligned[i] - pp_aligned[i]) * 0.02  # Within 2% of PP
            trend_fail = (position == 1 and close[i] < ema_20_aligned[i]) or \
                         (position == -1 and close[i] > ema_20_aligned[i])
            
            if tp_long or tp_short or at_pivot or trend_fail:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals