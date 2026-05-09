#!/usr/bin/env python3
# Hypothesis: 4h timeframe with 1-week EMA trend filter and 1-day Bollinger Band breakout.
# In strong weekly trends (price above/below weekly EMA), price tends to continue in that direction.
# Enter long when price breaks above 1-day Bollinger upper band in uptrend, short when breaks below lower band in downtrend.
# Uses Bollinger Band width contraction as volatility filter to avoid false breakouts in high volatility.
# Target: 20-50 total trades over 4 years (5-12/year) with size 0.25.

name = "4h_WeeklyEMA_Trend_BollingerBreakout"
timeframe = "4h"
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
    
    # Calculate 1-week EMA (50-period) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    ema_50 = close_1w.ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_values = ema_50.values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_values)
    
    # Determine trend: price above/below weekly EMA
    price_above_weekly_ema = close > ema_50_aligned
    price_below_weekly_ema = close < ema_50_aligned
    
    # Calculate 1-day Bollinger Bands (20, 2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean()
    std_20 = close_1d.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    upper_bb_values = upper_bb.values
    lower_bb_values = lower_bb.values
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_values)
    
    # Bollinger Band width for volatility filter (avoid false breakouts in high volatility)
    bb_width = upper_bb - lower_bb
    bb_width_ma = bb_width.rolling(window=20, min_periods=20).mean()
    bb_width_ratio = bb_width / bb_width_ma
    bb_width_ratio_values = bb_width_ratio.values
    bb_width_ratio_aligned = align_htf_to_ltf(prices, df_1d, bb_width_ratio_values)
    
    # Volatility filter: only allow breakouts when volatility is contracting or normal
    # (ratio < 1.5 means current width is less than 1.5x average width)
    volatility_filter = bb_width_ratio_aligned < 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(price_above_weekly_ema[i]) or np.isnan(price_below_weekly_ema[i]) or
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: uptrend (price > weekly EMA) + price breaks above BB upper + volatility filter
            if price_above_weekly_ema[i] and close[i] > upper_bb_aligned[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend (price < weekly EMA) + price breaks below BB lower + volatility filter
            elif price_below_weekly_ema[i] and close[i] < lower_bb_aligned[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or price re-enters Bollinger Bands
            if (not price_above_weekly_ema[i]) or (close[i] < sma_20.iloc[-1] if hasattr(sma_20, 'iloc') else close[i] < sma_20[-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or price re-enters Bollinger Bands
            if (not price_below_weekly_ema[i]) or (close[i] > sma_20.iloc[-1] if hasattr(sma_20, 'iloc') else close[i] > sma_20[-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals