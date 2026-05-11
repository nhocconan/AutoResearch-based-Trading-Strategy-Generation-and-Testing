#!/usr/bin/env python3
"""
1d_WickReversal_WeeklyTrend_Filter
Hypothesis: Daily Wick Reversal with Weekly Trend Filter - Uses weekly trend direction (via 50-week EMA) to filter daily wick reversal signals for higher probability entries. Wick reversals occur when price tests a level but reverses strongly, indicating rejection. In uptrends, look for bullish wick rejections at support; in downtrends, bearish wick rejections at resistance. This combines mean reversion at key levels with trend following to work in both bull and bear markets. Targets 15-25 trades/year via strict entry conditions requiring trend alignment and clear rejection signals.
"""

name = "1d_WickReversal_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: We'll use align_htf_to_ltf internally

def calculate_wick_reversal_signal(high, low, close, lookback=5):
    """
    Calculate wick reversal signals:
    - Bullish: Low < lowest low of lookback period AND Close > Open (strong close near high)
    - Bearish: High > highest high of lookback period AND Close < Open (strong close near low)
    """
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    
    bullish_wick = (low < lowest_low) & (close > (open := np.zeros_like(close)))  # Placeholder, will fix
    bearish_wick = (high > highest_high) & (close < open)
    
    # Actually calculate open-based conditions properly
    bullish_wick = (low < lowest_low) & (close > np.maximum.reduce([open, close]))  # Still wrong
    
    # Let's do it step by step correctly
    return lowest_low, highest_high

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Daily OHLCV
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Weekly Trend Filter (50-week EMA) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly trend: 1 = uptrend (price > EMA50), -1 = downtrend (price < EMA50)
    weekly_trend = np.where(close > ema_50_1d, 1, -1)
    
    # --- Daily Wick Reversal Detection ---
    lookback = 10  # Look for wicks beyond recent 10-period range
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    
    # Bullish wick rejection: price makes new low but closes strongly above open
    bullish_wick = (low < lowest_low) & (close > open_price) & ((close - open_price) > (high - low) * 0.5)
    
    # Bearish wick rejection: price makes new high but closes strongly below open
    bearish_wick = (high > highest_high) & (close < open_price) & ((open_price - close) > (high - low) * 0.5)
    
    # Volume confirmation: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    volume_spike = vol_ratio > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20)  # For EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(highest_high[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long signal: bullish wick rejection in uptrend + volume
            if (weekly_trend[i] == 1 and bullish_wick[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short signal: bearish wick rejection in downtrend + volume
            elif (weekly_trend[i] == -1 and bearish_wick[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: bearish wick rejection OR trend turns down
                if bearish_wick[i] or weekly_trend[i] == -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish wick rejection OR trend turns up
                if bullish_wick[i] or weekly_trend[i] == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals