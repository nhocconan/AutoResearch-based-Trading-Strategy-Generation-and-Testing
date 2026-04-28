# 12h Multi-Timeframe Bollinger Breakout Strategy
# Strategy Type: Bollinger Bands breakout with trend confirmation and volume filter
# Timeframe: 12h (primary), with 1d and 1w for higher timeframe context
# Why it should work in both bull and bear: 
# - Bollinger Bands adapt to volatility, expanding in trending markets and contracting in ranging markets
# - Breakouts capture momentum in trending markets while the trend filter avoids false signals in ranging markets
# - Volume confirmation ensures breakouts have conviction
# - The strategy is designed to capture significant moves while avoiding whipsaws in choppy markets
# - Target trades: 50-150 total over 4 years (12-37/year) to minimize fee drag

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
    
    # Get 1d data for Bollinger Bands and trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on daily data (20-period, 2 standard deviations)
    close_1d = df_1d['close'].values
    bb_period = 20
    bb_std = 2
    
    # Calculate middle band (SMA)
    sma_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    
    # Calculate standard deviation
    std_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    
    # Calculate upper and lower bands
    upper_bb = sma_1d + (bb_std * std_1d)
    lower_bb = sma_1d - (bb_std * std_1d)
    
    # Calculate 1d EMA for trend direction (50-period)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 1w EMA for higher timeframe trend (20-period)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align all indicators to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 0-23 UTC (trade all hours for 12h timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(sma_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: trade all hours for 12h timeframe (no restriction)
        # For 12h, we can trade any hour as each bar represents 12 hours
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Breakout conditions:
        # Long: price breaks above upper Bollinger Band with volume and trend alignment
        # Short: price breaks below lower Bollinger Band with volume and trend alignment
        long_breakout = close[i] > upper_bb_aligned[i]
        short_breakout = close[i] < lower_bb_aligned[i]
        
        # Trend filters:
        # For long: price above 1d EMA50 and 1w EMA20 (bullish alignment)
        # For short: price below 1d EMA50 and 1w EMA20 (bearish alignment)
        bullish_alignment = close[i] > ema_50_1d_aligned[i] and close[i] > ema_20_1w_aligned[i]
        bearish_alignment = close[i] < ema_50_1d_aligned[i] and close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_filter and bullish_alignment
        short_entry = short_breakout and vol_filter and bearish_alignment
        
        # Exit conditions: return to middle Bollinger Band
        long_exit = close[i] < sma_1d_aligned[i]
        short_exit = close[i] > sma_1d_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Bollinger_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0