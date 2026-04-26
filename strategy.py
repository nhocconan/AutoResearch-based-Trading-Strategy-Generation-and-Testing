#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_HTFRegime
Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R1 level AND 1d trend is up (close > EMA34) AND volume > 2.0x 20-period average AND 1w choppiness regime is trending (CHOP < 40). Enter short when price breaks below S1 level AND 1d trend is down (close < EMA34) AND volume spike AND 1w CHOP < 40. Uses tighter timeframe (12h) for lower trade frequency, Camarilla R1/S1 for precise intraday levels, 1d EMA34 for trend alignment, volume confirmation for institutional participation, and 1w choppiness filter to avoid ranging markets. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Daily Camarilla Pivot Levels (R1, S1)
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + ((high-low)*1.1/12), S1 = close - ((high-low)*1.1/12)
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 1w data for choppiness regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w Choppiness Index (CHOP < 40 = trending)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr_1w = np.maximum(np.maximum(high_1w - low_1w, np.abs(high_1w - np.roll(close_1w, 1))), np.abs(low_1w - np.roll(close_1w, 1)))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    atr_sum_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    
    # Max high and min low over 14 periods
    max_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl_1w = max_high_1w - min_low_1w
    range_hl_1w = np.maximum(range_hl_1w, 1e-10)
    
    # Chop calculation
    chop_1w = 100 * np.log10(atr_sum_1w / range_hl_1w) / np.log10(14)
    chop_trending_1w = chop_1w < 40.0  # Trending regime on weekly
    
    # Align 1w chop to 12h timeframe
    chop_trending_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_trending_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (34), volume MA warmup (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(chop_trending_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions relative to Camarilla levels
        breakout_above_r1 = close[i] > camarilla_r1_aligned[i]
        breakout_below_s1 = close[i] < camarilla_s1_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price above R1 + 1d uptrend + volume spike + 1w trending regime
            long_signal = breakout_above_r1 and trend_uptrend and volume_spike[i] and chop_trending_1w_aligned[i]
            
            # Short: price below S1 + 1d downtrend + volume spike + 1w trending regime
            short_signal = breakout_below_s1 and trend_downtrend and volume_spike[i] and chop_trending_1w_aligned[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend change to downtrend
            if close[i] < camarilla_s1_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend change to uptrend
            if close[i] > camarilla_r1_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_HTFRegime"
timeframe = "12h"
leverage = 1.0