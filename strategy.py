#!/usr/bin/env python3
"""
6h_Weekly_Pivot_PriceAction
Hypothesis: Weekly pivot levels act as strong support/resistance on 6h charts.
Breakouts above weekly R1 or below S1 with price action confirmation (engulfing candle)
and volume spike capture institutional moves. The weekly timeframe provides structural
levels that work in both bull and bear markets by adapting to the dominant trend.
Target: 15-25 trades/year to minimize fee drag.
"""

name = "6h_Weekly_Pivot_PriceAction"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # Resistance 1 (R1) = (2 * PP) - Low
    # Support 1 (S1) = (2 * PP) - High
    # Resistance 2 (R2) = PP + (High - Low)
    # Support 2 (S2) = PP - (High - Low)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = (2 * weekly_pp) - weekly_low
    weekly_s1 = (2 * weekly_pp) - weekly_high
    weekly_r2 = weekly_pp + (weekly_high - weekly_low)
    weekly_s2 = weekly_pp - (weekly_high - weekly_low)
    
    # Align weekly levels to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Price action: bullish engulfing pattern
    # Current candle engulfs previous candle's body
    bullish_engulfing = (close > open_prices) & (open_prices < close_prev) & (close > close_prev)
    bearish_engulfing = (close < open_prices) & (open_prices > close_prev) & (close < close_prev)
    
    # Need open and previous close
    open_prices = prices['open'].values
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]  # first bar
    
    bullish_engulfing = (close > open_prices) & (open_prices < close_prev) & (close > close_prev)
    bearish_engulfing = (close < open_prices) & (open_prices > close_prev) & (close < close_prev)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above weekly R1 with bullish engulfing and volume
            if (close[i] > weekly_r1_aligned[i] and 
                bullish_engulfing[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S1 with bearish engulfing and volume
            elif (close[i] < weekly_s1_aligned[i] and 
                  bearish_engulfing[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly PP or shows weakness
            if (close[i] < weekly_pp_aligned[i]) or \
               (close[i] < weekly_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly PP or shows strength
            if (close[i] > weekly_pp_aligned[i]) or \
               (close[i] > weekly_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals