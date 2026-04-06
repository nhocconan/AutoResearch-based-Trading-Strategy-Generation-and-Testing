#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Long when price breaks above Donchian high and 12h close > 12h open (bullish).
# Short when price breaks below Donchian low and 12h close < 12h open (bearish).
# Volume > 1.3x 20-period average for confirmation.
# Trend filter avoids counter-trend trades. Donchian breakouts capture momentum.
# Target: 100-200 total trades over 4 years (25-50/year).

name = "4h_donchian20_12h_trend_vol_v1"
timeframe = "4h"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 12h trend filter: bullish/bearish based on close vs open
    df_12h = get_htf_data(prices, '12h')
    twelve_h_open = df_12h['open'].values
    twelve_h_close = df_12h['close'].values
    twelve_h_bullish = twelve_h_close > twelve_h_open  # True for bullish 12h
    twelve_h_bearish = twelve_h_close < twelve_h_open   # True for bearish 12h
    twelve_h_bullish_aligned = align_htf_to_ltf(prices, df_12h, twelve_h_bullish)
    twelve_h_bearish_aligned = align_htf_to_ltf(prices, df_12h, twelve_h_bearish)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if 12h trend data not available
        if np.isnan(twelve_h_bullish_aligned[i]) or np.isnan(twelve_h_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or 12h turns bearish
            if (low[i] <= donchian_low[i] or 
                twelve_h_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or 12h turns bullish
            if (high[i] >= donchian_high[i] or 
                twelve_h_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and 12h trend filter
            if volume_filter:
                # Long: break above Donchian high during bullish 12h
                if (high[i] > donchian_high[i] and 
                    twelve_h_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: break below Donchian low during bearish 12h
                elif (low[i] < donchian_low[i] and 
                      twelve_h_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals