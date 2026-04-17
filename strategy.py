#!/usr/bin/env python3
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
    volume = prices['volume'].values
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # Calculate daily ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 200-day SMA for long-term trend filter
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(sma200[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0 * 20-day average volume (aligned)
        volume_filter = volume[i] > (2.0 * volume_ma20_1d_aligned[i])
        
        # Trend filter: price above 200-day SMA for long, below for short
        long_trend = close[i] > sma200[i]
        short_trend = close[i] < sma200[i]
        
        # Weekly trend filter: price above weekly EMA50 for long, below for short
        weekly_trend_long = close[i] > ema50_1w_aligned[i]
        weekly_trend_short = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price above both SMAs with volume confirmation
            if long_trend and weekly_trend_long and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below both SMAs with volume confirmation
            elif short_trend and weekly_trend_short and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 200-day SMA
            if close[i] < sma200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 200-day SMA
            if close[i] > sma200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA50_SMA200_VolumeFilter"
timeframe = "1d"
leverage = 1.0