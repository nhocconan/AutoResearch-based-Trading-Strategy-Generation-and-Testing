#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter
# Long when: price breaks above Donchian(20) high AND volume > 1.5 * volume MA(20) AND 1d chop > 61.8 (range market → mean reversion fade)
# Short when: price breaks below Donchian(20) low AND volume > 1.5 * volume MA(20) AND 1d chop > 61.8 (range market → mean reversion fade)
# Exit when: price returns to Donchian(20) midpoint OR chop < 38.2 (trending regime → stop fade)
# Uses 4h timeframe with 1d HTF for chop regime filter (target: 75-200 total over 4 years)
# Donchian breakouts capture momentum, volume confirmation avoids false breaks, chop regime ensures fading in ranging markets
# Works in both bull/bear by fading extremes in ranging conditions (chop > 61.8) and stopping when trend emerges (chop < 38.2)
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure

name = "4h_Donchian20_Volume_1dChopRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for chop calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index(14)
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(np.diff(high_1d))
        tr2 = np.abs(np.diff(low_1d))
        tr3 = np.abs(np.diff(close_1d))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # ATR(14) - smoothed TR
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = previous * (1 - 1/period) + current * (1/period)
            alpha = 1 / period
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
                else:
                    result[i] = result[i-1]
            return result
        
        atr = wilders_smoothing(tr, 14)
        
        # Highest high and lowest low over 14 periods
        highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Chop = 100 * log10(sum(ATR(14)) / (max(high)-min(low))) / log10(14)
        sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
        range_hl = highest_high - lowest_low
        
        # Avoid division by zero and log of zero
        chop = np.where((range_hl > 0) & (sum_atr > 0), 
                        100 * np.log10(sum_atr / range_hl) / np.log10(14), 
                        50)  # Default to neutral when invalid
    else:
        chop = np.full(len(close_1d), 50.0)  # Neutral default
    
    # Align 1d chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 4h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume confirmation AND chop > 61.8 (range market)
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * volume_ma[i] and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume confirmation AND chop > 61.8 (range market)
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * volume_ma[i] and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR chop < 38.2 (trending regime)
            if (close[i] >= donchian_mid[i] or chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR chop < 38.2 (trending regime)
            if (close[i] <= donchian_mid[i] or chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals