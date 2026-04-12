#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h price closing above/below 1d SMA50 with 12h trend filter and volume confirmation
    # Uses long-term trend (12h SMA50) to filter direction, 1d SMA50 as dynamic support/resistance
    # Volume confirmation ensures breakouts have participation. Works in bull/bear by
    # only taking trades aligned with higher timeframe trend. Target: 20-30 trades/year.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h SMA50 for trend filter
    sma_50_12h = np.full(len(df_12h), np.nan)
    for i in range(49, len(df_12h)):
        sma_50_12h[i] = np.mean(close_12h[i-49:i+1])
    sma_50_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_50_12h)
    
    # Get 1d data for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d SMA50 as dynamic support/resistance
    sma_50_1d = np.full(len(df_1d), np.nan)
    for i in range(49, len(df_1d)):
        sma_50_1d[i] = np.mean(close_1d[i-49:i+1])
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(sma_50_12h_aligned[i]) or np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price should be on same side of 12h SMA50 as 1d SMA50
        trend_bullish = close[i] > sma_50_12h_aligned[i]
        trend_bearish = close[i] < sma_50_12h_aligned[i]
        
        # Position relative to 1d SMA50
        price_above_1d_sma = close[i] > sma_50_1d_aligned[i]
        price_below_1d_sma = close[i] < sma_50_1d_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = trend_bullish and price_above_1d_sma and vol_filter[i]
        short_entry = trend_bearish and price_below_1d_sma and vol_filter[i]
        
        # Exit conditions: trend reversal or price crosses back below/above 1d SMA50
        long_exit = (not trend_bullish) or (price_below_1d_sma)
        short_exit = (not trend_bearish) or (price_above_1d_sma)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_1d_sma50_trend_filter_volume"
timeframe = "4h"
leverage = 1.0