#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1-week trend filter (EMA20) and Bollinger Band squeeze breakout.
# Long: Price breaks above upper Bollinger Band (20,2) during low volatility (BB width < 50th percentile) + weekly EMA20 uptrend.
# Short: Price breaks below lower Bollinger Band (20,2) during low volatility + weekly EMA20 downtrend.
# Uses Bollinger squeeze to identify low volatility breakouts, weekly EMA for trend filter.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = np.full(len(close_1d), np.nan)
    std = np.full(len(close_1d), np.nan)
    upper = np.full(len(close_1d), np.nan)
    lower = np.full(len(close_1d), np.nan)
    bb_width = np.full(len(close_1d), np.nan)
    
    for i in range(bb_period - 1, len(close_1d)):
        sma[i] = np.mean(close_1d[i - bb_period + 1:i + 1])
        std[i] = np.std(close_1d[i - bb_period + 1:i + 1])
        upper[i] = sma[i] + bb_std * std[i]
        lower[i] = sma[i] - bb_std * std[i]
        bb_width[i] = (upper[i] - lower[i]) / sma[i] * 100  # Percentage width
    
    # Weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20 = np.full(len(close_1w), np.nan)
    multiplier = 2 / (20 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_20[i] = close_1w[i]
        elif np.isnan(ema_20[i-1]):
            ema_20[i] = close_1w[i]
        else:
            ema_20[i] = (close_1w[i] - ema_20[i-1]) * multiplier + ema_20[i-1]
    
    # Align indicators to 1d
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Percentile rank of BB width (50-period lookback)
    bb_width_percentile = np.full(n, np.nan)
    lookback = 50
    for i in range(lookback, n):
        if not np.isnan(bb_width_aligned[i]):
            window = bb_width_aligned[i - lookback:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                bb_width_percentile[i] = (np.sum(valid < bb_width_aligned[i]) / len(valid)) * 100
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(lookback, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(ema_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_percent = bb_width_percentile[i]
        ema_trend = ema_20_aligned[i]
        
        # Squeeze condition: BB width below 50th percentile (low volatility)
        squeeze = bb_percent < 50
        
        if position == 0:
            # Long: price breaks above upper BB during squeeze + weekly uptrend
            if price > upper_aligned[i] and squeeze and price > ema_trend:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower BB during squeeze + weekly downtrend
            elif price < lower_aligned[i] and squeeze and price < ema_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower BB
            if price < lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper BB
            if price > upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Bollinger_Squeeze_EMA20"
timeframe = "1d"
leverage = 1.0