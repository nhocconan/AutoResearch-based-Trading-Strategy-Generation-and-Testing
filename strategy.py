#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) AND weekly close > weekly EMA50 (bullish trend) AND daily volume > 1.5x 20-day average.
# Short when Williams %R crosses below -80 (overbought rejection) AND weekly close < weekly EMA50 (bearish trend) AND daily volume > 1.5x 20-day average.
# Exit when Williams %R returns to -50 (mean reversion center).
# Uses 1d timeframe with 1w trend filter and daily volume confirmation.
# Target: 20-60 total trades over 4 years (5-15/year) to avoid fee drag.

name = "1d_WilliamsR_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Daily data for volume
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Williams %R(14) on 1d data
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Avoid division by zero
    
    # Weekly EMA50 for trend filter
    weekly_close = df_w['close'].values
    ema50_w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_trend_up = weekly_close > ema50_w
    weekly_trend_down = weekly_close < ema50_w
    trend_up = align_htf_to_ltf(prices, df_w, weekly_trend_up)
    trend_down = align_htf_to_ltf(prices, df_w, weekly_trend_down)
    
    # Daily volume filter: current volume > 1.5x 20-day average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.5 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(williams_period, 20, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -20 from below, bullish trend, volume confirmation
            if (williams_r[i] > -20 and williams_r[i-1] <= -20 and 
                trend_up[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from above, bearish trend, volume confirmation
            elif (williams_r[i] < -80 and williams_r[i-1] >= -80 and 
                  trend_down[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion)
            if williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion)
            if williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals