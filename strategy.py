#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Long when price breaks above Donchian upper band during bullish 12h trend with volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band during bearish 12h trend with volume confirmation.
# Uses 12h trend filter to avoid counter-trend trades. Donchian provides clear breakout points.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within optimal range.

name = "4h_donchian20_12h_trend_vol_v1"
timeframe = "4h"
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
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 12h trend filter: bullish/bearish based on close vs open
    df_12h = get_htf_data(prices, '12h')
    trend_open = df_12h['open'].values
    trend_close = df_12h['close'].values
    trend_bullish = trend_close > trend_open  # True for bullish trend
    trend_bearish = trend_close < trend_open   # True for bearish trend
    trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, trend_bullish)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, trend_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if trend data not available
        if np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below Donchian lower or trend turns bearish
            if (low[i] <= donchian_low[i] or 
                trend_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above Donchian upper or trend turns bullish
            if (high[i] >= donchian_high[i] or 
                trend_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: break above Donchian upper during bullish trend
                if (high[i] > donchian_high[i] and 
                    trend_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below Donchian lower during bearish trend
                elif (low[i] < donchian_low[i] and 
                      trend_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals