# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Choppy_Market_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === DAILY CALCULATIONS ===
    # Donchian breakout levels (20-period)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Chopiness Index for regime detection
    atr_series = pd.Series(
        np.maximum(
            np.maximum(df_1d['high'] - df_1d['low'],
                      np.abs(df_1d['high'] - df_1d['close'].shift(1))),
            np.abs(df_1d['low'] - df_1d['close'].shift(1))
        )
    )
    atr_sum = atr_series.rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.values
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_filter = df_1d['volume'].values > (vol_ma * 1.5)
    
    # === WEEKLY TREND FILTER ===
    # Weekly EMA34 for trend direction
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === ALIGN TO DAILY TIMEFRAME ===
    donchian_high_d = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_d = align_htf_to_ltf(prices, df_1d, donchian_low)
    chop_d = align_htf_to_ltf(prices, df_1d, chop_values)
    volume_filter_d = align_htf_to_ltf(prices, df_1d, volume_filter)
    ema34_1w_d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # Need enough data for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(donchian_high_d[i]) or np.isnan(donchian_low_d[i]) or
            np.isnan(chop_d[i]) or np.isnan(volume_filter_d[i]) or
            np.isnan(ema34_1w_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop regime: > 61.8 = ranging (mean reversion), < 38.2 = trending
        is_ranging = chop_d[i] > 61.8
        is_trending = chop_d[i] < 38.2
        
        if position == 0:
            # In ranging market: mean reversion at Donchian bands
            if is_ranging:
                # Long at lower band with volume confirmation
                if close[i] <= donchian_low_d[i] and volume_filter_d[i]:
                    signals[i] = 0.25
                    position = 1
                # Short at upper band with volume confirmation
                elif close[i] >= donchian_high_d[i] and volume_filter_d[i]:
                    signals[i] = -0.25
                    position = -1
            # In trending market: breakout in direction of weekly trend
            elif is_trending:
                weekly_uptrend = close[i] > ema34_1w_d[i]
                weekly_downtrend = close[i] < ema34_1w_d[i]
                
                # Long breakout in uptrend
                if close[i] > donchian_high_d[i] and weekly_uptrend and volume_filter_d[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown in downtrend
                elif close[i] < donchian_low_d[i] and weekly_downtrend and volume_filter_d[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if is_ranging:
                # In ranging market: exit at opposite band
                if close[i] >= donchian_high_d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In trending market: trail with weekly trend
                if close[i] < ema34_1w_d[i]:  # Trend reversal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if is_ranging:
                # In ranging market: exit at opposite band
                if close[i] <= donchian_low_d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In trending market: trail with weekly trend
                if close[i] > ema34_1w_d[i]:  # Trend reversal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals