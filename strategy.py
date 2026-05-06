#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Bands with volume confirmation and trend filter
# Weekly Bollinger Bands (20,2) on 1w data provide dynamic support/resistance levels
# Price touching upper band with volume > 1.5x 20-day average and price > 50-day EMA = short signal
# Price touching lower band with volume > 1.5x 20-day average and price < 50-day EMA = long signal
# Trend filter: 50-day EMA determines bias (only long when above EMA, short when below)
# Volume confirmation reduces false signals in low-volume environments
# Works in bull/bear markets: mean reversion at extremes with trend filter prevents counter-trend trades
# Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_WeeklyBB_Touch_VolumeTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Bollinger Bands ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly close for Bollinger Bands
    weekly_close = df_1w['close'].values
    
    # Bollinger Bands (20,2)
    bb_middle = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    
    # Align weekly Bollinger Bands to daily timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    
    # Volume confirmation: >1.5x 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Trend filter: 50-day EMA
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: price touches lower BB AND volume confirmation AND price < EMA50 (oversold in downtrend)
            if low[i] <= bb_lower_aligned[i] and volume_filter[i] and close[i] < ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short signal: price touches upper BB AND volume confirmation AND price > EMA50 (overbought in uptrend)
            elif high[i] >= bb_upper_aligned[i] and volume_filter[i] and close[i] > ema_50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above middle band or reaches upper band (mean reversion complete)
            if close[i] > bb_middle[-1] if not np.isnan(bb_middle[-1]) else close[i] > bb_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below middle band or reaches lower band (mean reversion complete)
            if close[i] < bb_middle[-1] if not np.isnan(bb_middle[-1]) else close[i] < bb_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals