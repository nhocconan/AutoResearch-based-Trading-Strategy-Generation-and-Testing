#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1d trend filter and volume spike
# BB breakouts capture momentum, filtered by 1d EMA trend to avoid counter-trend trades.
# Volume spike confirms institutional participation. Works in bull/bear by aligning
# breakout direction with higher timeframe trend. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2) on 1d close
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean()
    std_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std()
    upper_bb = sma_20 + (std_20 * bb_std)
    lower_bb = sma_20 - (std_20 * bb_std)
    
    # Align Bollinger Bands to 6h timeframe (wait for 1d close)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb.values)
    
    # 1d EMA trend filter (50-period)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need BB (20), EMA (50), volume MA (20)
    start_idx = max(bb_period, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: break above upper BB with volume and bullish trend
            if price > upper_bb_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: break below lower BB with volume and bearish trend
            elif price < lower_bb_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB or trend turns bearish
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if price <= middle_bb or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle BB or trend turns bullish
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if price >= middle_bb or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Bollinger_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0