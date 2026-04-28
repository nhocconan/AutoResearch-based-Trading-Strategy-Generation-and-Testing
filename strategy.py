# 4h_Macd_Histogram_Zero_Cross_Trend_Filter
# Hypothesis: MACD histogram crossing zero with trend filter (EMA50) and volume confirmation captures sustained momentum moves in both bull and bear markets.
# The MACD histogram crossing zero indicates a change in short-term momentum relative to longer-term trend, which when aligned with the EMA50 trend and confirmed by volume, provides high-probability entries.
# Target: 20-40 trades/year on 4h timeframe to minimize fee drag while capturing significant moves.
# Uses EMA50 for trend filter and volume > 20-period average for confirmation.

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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate MACD (12,26,9) on close prices
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Align daily EMA50 to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(macd_hist[i]) or
            np.isnan(macd_hist[i-1]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # MACD histogram zero cross signals
        macd_bullish_cross = macd_hist[i] > 0 and macd_hist[i-1] <= 0
        macd_bearish_cross = macd_hist[i] < 0 and macd_hist[i-1] >= 0
        
        long_entry = macd_bullish_cross and uptrend and vol_filter
        short_entry = macd_bearish_cross and downtrend and vol_filter
        
        # Exit when MACD histogram crosses back in opposite direction or trend fails
        macd_bearish_exit = macd_hist[i] < 0 and macd_hist[i-1] >= 0
        macd_bullish_exit = macd_hist[i] > 0 and macd_hist[i-1] <= 0
        
        long_exit = macd_bearish_exit or not uptrend
        short_exit = macd_bullish_exit or not downtrend
        
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
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Macd_Histogram_Zero_Cross_Trend_Filter"
timeframe = "4h"
leverage = 1.0