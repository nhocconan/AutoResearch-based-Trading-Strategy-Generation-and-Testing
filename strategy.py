#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d Trend-Following with Volume Confirmation and Session Filter
# Uses 4h EMA20 and 1d EMA50 for trend alignment (bullish if price > both EMA, bearish if price < both)
# Entry on 1h breakout of 20-bar high/low with volume > 1.5x 20-period average
# Only trades during 08-20 UTC to avoid low-liquidity periods
# Fixed position size of 0.20 to control risk and minimize fee churn
# Designed for 15-30 trades/year to avoid fee drag while capturing trends
name = "1h_4h1d_EMA_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA20 for short-term trend
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1h = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1d EMA50 for long-term trend
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period high/low for breakout detection
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1h[i]) or np.isnan(ema50_1h[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: Price breaks above 20-period high, above both EMAs, volume spike, in session
            if (close[i] > high_max[i] and 
                close[i] > ema20_1h[i] and 
                close[i] > ema50_1h[i] and 
                vol_spike and 
                in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 20-period low, below both EMAs, volume spike, in session
            elif (close[i] < low_min[i] and 
                  close[i] < ema20_1h[i] and 
                  close[i] < ema50_1h[i] and 
                  vol_spike and 
                  in_session[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below 20-period low OR below 4h EMA20
            if close[i] < low_min[i] or close[i] < ema20_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price rises back above 20-period high OR above 4h EMA20
            if close[i] > high_max[i] or close[i] > ema20_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals