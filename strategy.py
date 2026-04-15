# 2025-07-01: 6h_WeeklyPivot_Donchian_Trend_Filter
# Hypothesis: 6h Donchian breakout with weekly pivot direction filter and volume confirmation.
# In bull markets: long when price > weekly pivot and breaks Donchian(20) high with volume.
# In bear markets: short when price < weekly pivot and breaks Donchian(20) low with volume.
# Weekly pivot provides structural bias, reducing false breakouts in chop.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Uses 6h timeframe with weekly and daily HTF for pivot and trend context.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR WEEKLY PIVOT (using 1d as proxy for weekly pivot calc) ===
    # Note: For true weekly pivot, we would use 1w data, but 1d OHLC can approximate
    # weekly pivot by using weekly resample logic conceptually.
    # However, per rules we must use actual HTF data - we'll use 1d for pivot components
    # and align properly. For true weekly pivot, we need 1w data.
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot from 1w OHLC
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    if len(df_1w) > 0:
        wk_high = df_1w['high'].values
        wk_low = df_1w['low'].values
        wk_close = df_1w['close'].values
        wk_pivot = (wk_high + wk_low + wk_close) / 3.0
        wk_range = wk_high - wk_low
        wk_r1 = 2 * wk_pivot - wk_low
        wk_s1 = 2 * wk_pivot - wk_high
        wk_r2 = wk_pivot + wk_range
        wk_s2 = wk_pivot - wk_range
        
        # Align weekly pivot levels to 6h timeframe (wait for weekly close)
        wk_pivot_aligned = align_htf_to_ltf(prices, df_1w, wk_pivot)
        wk_r1_aligned = align_htf_to_ltf(prices, df_1w, wk_r1)
        wk_s1_aligned = align_htf_to_ltf(prices, df_1w, wk_s1)
        wk_r2_aligned = align_htf_to_ltf(prices, df_1w, wk_r2)
        wk_s2_aligned = align_htf_to_ltf(prices, df_1w, wk_s2)
    else:
        wk_pivot_aligned = np.full(n, np.nan)
        wk_r1_aligned = np.full(n, np.nan)
        wk_s1_aligned = np.full(n, np.nan)
        wk_r2_aligned = np.full(n, np.nan)
        wk_s2_aligned = np.full(n, np.nan)
    
    # === DAILY TREND FILTER (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) > 0:
        ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 6H DONCHIAN CHANNEL (20) ===
    # Use 6h high/low for Donchian
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # === VOLUME FILTER ===
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median  # Volume > 1.5x median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(wk_pivot_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_threshold[i])):
            signals[i] = signals[i-1] if i > 0 else 0
            continue
        
        # Long conditions:
        # 1. Price above weekly pivot (bullish bias)
        # 2. Break above Donchian high with volume confirmation
        # 3. Above daily EMA50 (additional trend filter)
        long_condition = (close[i] > wk_pivot_aligned[i] and 
                         close[i] > donchian_high[i] and 
                         volume[i] > vol_threshold[i] and
                         close[i] > ema_50_1d_aligned[i])
        
        # Short conditions:
        # 1. Price below weekly pivot (bearish bias)
        # 2. Break below Donchian low with volume confirmation
        # 3. Below daily EMA50 (additional trend filter)
        short_condition = (close[i] < wk_pivot_aligned[i] and 
                          close[i] < donchian_low[i] and 
                          volume[i] > vol_threshold[i] and
                          close[i] < ema_50_1d_aligned[i])
        
        if long_condition:
            signals[i] = 0.25
        elif short_condition:
            signals[i] = -0.25
        # Exit conditions: price returns inside Donchian channel
        elif i > 0 and signals[i-1] != 0:
            if signals[i-1] == 0.25 and close[i] < donchian_high[i]:
                signals[i] = 0.0
            elif signals[i-1] == -0.25 and close[i] > donchian_low[i]:
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1] if i > 0 else 0
    
    return signals

name = "6h_WeeklyPivot_Donchian_Trend_Filter"
timeframe = "6h"
leverage = 1.0