#!/usr/bin/env python3
"""
6h_OrderBlock_Equilibrium_WeeklyTrend_v1
Hypothesis: Institutional order blocks (OB) act as support/resistance. In weekly uptrend, buy at bullish OB when price tests it; in weekly downtrend, sell at bearish OB when price tests it. Equilibrium (midpoint of daily range) filters for mean-reversion within the OB zone. Weekly trend avoids counter-trend trades. Target: 15-30 trades/year on 6h.
"""

name = "6h_OrderBlock_Equilibrium_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1W Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === 1D Data for Order Blocks and Equilibrium ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Bullish OB: down candle (close < open) followed by up candle -> OB = [low, high] of down candle
    # Bearish OB: up candle (close > open) followed by down candle -> OB = [low, high] of up candle
    open_1d = df_1d['open'].values
    
    bullish_ob_high = np.full(len(high_1d), np.nan)
    bullish_ob_low = np.full(len(high_1d), np.nan)
    bearish_ob_high = np.full(len(high_1d), np.nan)
    bearish_ob_low = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        # Bullish OB: red candle then green candle
        if close_1d[i-1] < open_1d[i-1] and close_1d[i] > open_1d[i]:
            bullish_ob_high[i] = high_1d[i-1]
            bullish_ob_low[i] = low_1d[i-1]
        # Bearish OB: green candle then red candle
        if close_1d[i-1] > open_1d[i-1] and close_1d[i] < open_1d[i]:
            bearish_ob_high[i] = high_1d[i-1]
            bearish_ob_low[i] = low_1d[i-1]
    
    # Forward-fill OBs to keep them active until broken
    # Convert to pandas Series for ffill, then back to array
    bullish_ob_high_series = pd.Series(bullish_ob_high).ffill()
    bullish_ob_low_series = pd.Series(bullish_ob_low).ffill()
    bearish_ob_high_series = pd.Series(bearish_ob_high).ffill()
    bearish_ob_low_series = pd.Series(bearish_ob_low).ffill()
    
    bullish_ob_high = bullish_ob_high_series.values
    bullish_ob_low = bullish_ob_low_series.values
    bearish_ob_high = bearish_ob_high_series.values
    bearish_ob_low = bearish_ob_low_series.values
    
    # Equilibrium: midpoint of daily range
    equilibrium_1d = (high_1d + low_1d) / 2.0
    
    # Align 1D data to 6h timeframe
    bullish_ob_high_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_high)
    bullish_ob_low_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_low)
    bearish_ob_high_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_high)
    bearish_ob_low_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_low)
    equilibrium_aligned = align_htf_to_ltf(prices, df_1d, equilibrium_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(bullish_ob_high_aligned[i]) or 
            np.isnan(bearish_ob_low_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: weekly uptrend + price tests bullish OB (within OB zone) + price > equilibrium (bullish bias)
            if (weekly_uptrend and 
                not np.isnan(bullish_ob_high_aligned[i]) and 
                not np.isnan(bullish_ob_low_aligned[i]) and
                low[i] <= bullish_ob_high_aligned[i] and  # price touches or penetrates OB high
                high[i] >= bullish_ob_low_aligned[i] and  # price touches or penetrates OB low
                close[i] > equilibrium_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price tests bearish OB + price < equilibrium (bearish bias)
            elif (weekly_downtrend and 
                  not np.isnan(bearish_ob_high_aligned[i]) and 
                  not np.isnan(bearish_ob_low_aligned[i]) and
                  low[i] <= bearish_ob_high_aligned[i] and
                  high[i] >= bearish_ob_low_aligned[i] and
                  close[i] < equilibrium_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below equilibrium OR weekly trend turns down
            if close[i] < equilibrium_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above equilibrium OR weekly trend turns up
            if close[i] > equilibrium_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals