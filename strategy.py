#!/usr/bin/env python3
"""
6h_WeeklyPivot_Position_Action
Hypothesis: Weekly pivot levels (PP, R1-R4, S1-S4) on 6h timeframe with price action confirmation.
In bull markets: buy near S1/S2 with bullish rejection candles. In bear markets: sell near R1/R2 with bearish rejection.
Uses 1d trend filter (EMA50) to avoid counter-trend trades and volume spike for confirmation.
Targets 15-30 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points from previous week's OHLC"""
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

def is_bullish_rejection(open_price, high, low, close):
    """Bullish rejection: long lower shadow, close near high"""
    body = abs(close - open_price)
    lower_shadow = min(open_price, close) - low
    upper_shadow = high - max(open_price, close)
    return lower_shadow > 2 * body and close > open_price

def is_bearish_rejection(open_price, high, low, close):
    """Bearish rejection: long upper shadow, close near low"""
    body = abs(close - open_price)
    lower_shadow = min(open_price, close) - low
    upper_shadow = high - max(open_price, close)
    return upper_shadow > 2 * body and close < open_price

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_weekly_pivot(
        weekly_high, weekly_low, weekly_close
    )
    
    # Align weekly pivot levels to 6h timeframe (previous week's levels)
    pp_6h = align_htf_to_ltf(prices, df_1w, pp)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    r4_6h = align_htf_to_ltf(prices, df_1w, r4)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 1.5x 24-period average (4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma_24 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Price near pivot levels (within 0.5%)
        near_s1 = abs(close[i] - s1_6h[i]) / s1_6h[i] < 0.005
        near_s2 = abs(close[i] - s2_6h[i]) / s2_6h[i] < 0.005
        near_r1 = abs(close[i] - r1_6h[i]) / r1_6h[i] < 0.005
        near_r2 = abs(close[i] - r2_6h[i]) / r2_6h[i] < 0.005
        
        # Trend filter from daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Price action confirmation
        bullish_rej = is_bullish_rejection(open_price[i], high[i], low[i], close[i])
        bearish_rej = is_bearish_rejection(open_price[i], high[i], low[i], close[i])
        
        # Entry conditions
        long_entry = (near_s1 or near_s2) and volume_spike[i] and uptrend and bullish_rej
        short_entry = (near_r1 or near_r2) and volume_spike[i] and downtrend and bearish_rej
        
        # Exit on opposite signal or stop at opposite pivot
        long_exit = (near_r1 or near_r2) and volume_spike[i]
        short_exit = (near_s1 or near_s2) and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Position_Action"
timeframe = "6h"
leverage = 1.0