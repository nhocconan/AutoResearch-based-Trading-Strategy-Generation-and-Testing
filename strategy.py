#!/usr/bin/env python3
# 1D_Wilson_Triple_Band_Squeeze_v1
# Hypothesis: Combines Bollinger Band squeeze detection with 20-period SMA direction and volume confirmation on daily timeframe to capture breakout trades in both trending and ranging markets. The squeeze identifies low volatility periods preceding explosive moves, while SMA direction filters for trend alignment. Volume confirmation ensures institutional participation. Designed for low frequency (10-25 trades/year) with high win rate potential in both bull and bear markets.

name = "1D_Wilson_Triple_Band_Squeeze_v1"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) for squeeze detection
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    bb_width = (upper_band - lower_band) / sma20
    
    # Bollinger Band squeeze: width below 50th percentile of last 50 days
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).rank(pct=True).values
    squeeze_condition = bb_width_percentile < 0.5
    
    # Weekly trend filter: price above/below 21-period EMA
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure we have BB width percentile, volume MA, and weekly EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(ema21_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: BB squeeze release + price above SMA20 + weekly uptrend + volume
            if (squeeze_condition[i] and 
                close[i] > sma20[i] and 
                close[i] > ema21_1w_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze release + price below SMA20 + weekly downtrend + volume
            elif (squeeze_condition[i] and 
                  close[i] < sma20[i] and 
                  close[i] < ema21_1w_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below SMA20 (mean reversion)
            if close[i] <= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above SMA20 (mean reversion)
            if close[i] >= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals