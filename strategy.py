#!/usr/bin/env python3
# Hypothesis: 1d timeframe with weekly Bollinger Band squeeze and breakout detection.
# Uses weekly Bollinger Bands to identify low volatility periods (squeeze) and breakouts.
# Weekly Bollinger Band width percentile < 20% indicates squeeze, then breakout above/below bands.
# Daily trend filter (EMA50) ensures trades align with higher timeframe trend.
# Volume confirmation reduces false breakouts.
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25.

name = "1d_Bollinger_Squeeze_Breakout_WeeklyEMA50_Trend_Volume"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20, 2)
    weekly_close = df_1w['close'].values
    sma_20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / sma_20  # Normalized width
    
    # Bollinger Band squeeze: width below 20th percentile (low volatility)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze = bb_width < bb_width_percentile
    
    # Breakout conditions: price breaks above upper band or below lower band
    breakout_up = weekly_close > upper_band
    breakout_down = weekly_close < lower_band
    
    # Align weekly indicators to daily timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze)
    breakout_up_aligned = align_htf_to_ltf(prices, df_1w, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1w, breakout_down)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(squeeze_aligned[i]) or np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or np.isnan(trend_up[i]) or 
            np.isnan(trend_down[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze breakout up + daily uptrend + volume spike
            if squeeze_aligned[i] and breakout_up_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down + daily downtrend + volume spike
            elif squeeze_aligned[i] and breakout_down_aligned[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly middle band or trend reversal
            weekly_sma_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
            if close[i] <= weekly_sma_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly middle band or trend reversal
            weekly_sma_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
            if close[i] >= weekly_sma_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals