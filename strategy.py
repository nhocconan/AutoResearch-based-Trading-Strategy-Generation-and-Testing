#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with weekly trend filter and volume confirmation.
# Long when W%R crosses above -50 (momentum shift) during bullish week with volume > 1.4x 24-period average.
# Short when W%R crosses below -50 during bearish week with volume confirmation.
# Uses weekly trend to avoid counter-trend trades. Williams %R provides timely momentum signals.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_williamsr_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    willr = willr.values  # Convert to numpy array
    
    # Weekly trend filter: bullish/bearish week based on close vs open
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish week
    weekly_bearish = weekly_close < weekly_open   # True for bearish week
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # Volume filter: current volume > 1.4x 24-period average (4 days)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if weekly trend data not available
        if np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.4
        
        # Williams %R crossover signals
        willr_cross_up = (willr[i] > -50) and (willr[i-1] <= -50)  # Cross above -50
        willr_cross_down = (willr[i] < -50) and (willr[i-1] >= -50)  # Cross below -50
        
        # Check exits
        if position == 1:  # long position
            # Exit: W%R crosses below -50 or weekly turn bearish
            if willr_cross_down or weekly_bearish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: W%R crosses above -50 or weekly turn bullish
            if willr_cross_up or weekly_bullish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            if volume_filter:
                # Long: W%R crosses above -50 during bullish week
                if willr_cross_up and weekly_bullish_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: W%R crosses below -50 during bearish week
                elif willr_cross_down and weekly_bearish_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals