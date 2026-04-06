#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) with 1d trend filter and volume confirmation.
# Long when Williams %R crosses above -20 during bullish day with volume > 1.2x 20-period average.
# Short when Williams %R crosses below -80 during bearish day with volume confirmation.
# Uses daily trend filter to avoid counter-trend trades. Williams %R identifies overbought/oversold conditions.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_williamsr_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Skip if Williams %R is not available
        if np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.2
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R drops below -50 or daily turn bearish
            if (williams_r[i] < -50 or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R rises above -50 or daily turn bullish
            if (williams_r[i] > -50 or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Long: Williams %R crosses above -20 during bullish day
                if (williams_r[i] > -20 and williams_r[i-1] <= -20 and 
                    daily_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -80 during bearish day
                elif (williams_r[i] < -80 and williams_r[i-1] >= -80 and 
                      daily_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals