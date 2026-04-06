#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Long when price breaks above cloud during bullish day (close > open) with volume > 1.5x 20-period average.
# Short when price breaks below cloud during bearish day (close < open) with volume confirmation.
# Ichimoku provides dynamic support/resistance; daily trend filter avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.

name = "6h_ichimoku_1d_trend_vol_v1"
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
    
    # Ichimoku Cloud components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    high26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    high52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high52 + low52) / 2
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Need 52 periods for Senkou B
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below Senkou Span B or daily turn bearish
            if (low[i] < senkou_b[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above Senkou Span A or daily turn bullish
            if (high[i] > senkou_a[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Long: price breaks above Senkou Span A (cloud top) during bullish day
                if (high[i] > senkou_a[i] and 
                    daily_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Senkou Span B (cloud bottom) during bearish day
                elif (low[i] < senkou_b[i] and 
                      daily_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals