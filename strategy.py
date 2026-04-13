#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h Bollinger Band mean reversion + 1d trend filter
# Strategy: Long when price touches lower 4h Bollinger Band (20,2) and closes above open (bullish candle)
#           Short when price touches upper 4h Bollinger Band (20,2) and closes below open (bearish candle)
#           Only in direction of 1d EMA50 trend (price > EMA50 for longs, price < EMA50 for shorts)
#           Uses Bollinger Bands for mean reversion in ranging markets and EMA for trend filter
#           Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
#           Session filter: 08-20 UTC to avoid low-volume Asian session

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Bollinger Bands (20,2)
    sma_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb_4h = sma_20_4h + 2 * std_20_4h
    lower_bb_4h = sma_20_4h - 2 * std_20_4h
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 1h timeframe
    upper_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_bb_4h)
    lower_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_bb_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% of capital
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(upper_bb_4h_aligned[i]) or 
            np.isnan(lower_bb_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger Band touch conditions
        bb_touch_low = low[i] <= lower_bb_4h_aligned[i]
        bb_touch_high = high[i] >= upper_bb_4h_aligned[i]
        
        # Candlestick confirmation
        bullish_candle = close[i] > open_prices[i]
        bearish_candle = close[i] < open_prices[i]
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic
        long_entry = bb_touch_low and bullish_candle and uptrend
        short_entry = bb_touch_high and bearish_candle and downtrend
        
        # Exit conditions: opposite BB touch or trend change
        exit_long = position == 1 and (bb_touch_high or not uptrend)
        exit_short = position == -1 and (bb_touch_low or not downtrend)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_bollinger_mean_reversion_trend_filter_v1"
timeframe = "1h"
leverage = 1.0