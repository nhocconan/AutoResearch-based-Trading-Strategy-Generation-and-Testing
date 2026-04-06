#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d Trend Filter and Volume Confirmation
Hypothesis: Ichimoku system (Tenkan/Kijun cross + cloud filter) on 6h with 1d trend alignment and volume confirmation provides robust signals in both bull and bear markets. The cloud acts as dynamic support/resistance, while TK cross captures momentum. Volume ensures institutional participation. Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend alignment (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for signals to avoid look-ahead
    
    # Cloud: between Senkou A and Senkou B
    # For signal generation, we use current Senkou spans (not shifted) to avoid look-ahead
    # Actually, for determining if price is above/below cloud, we need current cloud
    # The cloud plotted ahead is Senkou A/B shifted 26 periods, but current cloud uses unshifted
    # So we calculate the current cloud boundaries
    senkou_a_current = (tenkan + kijun) / 2  # This is actually Tenkan/Kijun average, not shifted
    senkou_b_current = (high_52 + low_52) / 2  # This is the 52-period midpoint
    
    # Actually, proper Ichimoku cloud calculation:
    # Senkou Span A = (Tenkan + Kijun)/2 plotted 26 periods ahead
    # Senkou Span B = (52-period high + 52-period low)/2 plotted 26 periods ahead
    # To check if price is above/cloud now, we compare price to the Senkou A/B values that were plotted 26 periods ago
    # So we need Senkou A/B values shifted BACK by 26 periods to align with current price
    
    senkou_a_shifted = np.roll(senkou_a, 26)  # Shifted back to align with current
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # 6h volume filter
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 4 days average
    vol_filter = volume > (1.3 * vol_ma)  # Require above-average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from sufficient warmup
    start = 100  # Ensures Ichimoku components are valid
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend: price above/below cloud + TK cross
        # Price above cloud: close > both Senkou A and Senkou B (current cloud)
        price_above_cloud = (close[i] > senkou_a_shifted[i]) and (close[i] > senkou_b_shifted[i])
        price_below_cloud = (close[i] < senkou_a_shifted[i]) and (close[i] < senkou_b_shifted[i])
        
        # TK cross: Tenkan crosses above/below Kijun
        tk_cross_up = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
        tk_cross_down = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
        
        # 1d trend filter
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price falls below cloud OR TK cross down OR 1d trend turns down
            if (price_below_cloud or tk_cross_down or not uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above cloud OR TK cross up OR 1d trend turns up
            if (price_above_cloud or tk_cross_up or not downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross + cloud filter + 1d trend alignment + volume
            long_setup = tk_cross_up and price_above_cloud and uptrend_1d and vol_filter[i]
            short_setup = tk_cross_down and price_below_cloud and downtrend_1d and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals