#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ChopFilter
Hypothesis: On 12h timeframe, use Camarilla R1/S1 levels from 1d for breakout entries, filtered by 1d trend direction (close > EMA34), volume spike (>2.0x 20-period average), and choppiness regime (CHOP > 61.8 = range -> mean reversion fade at S1/R1; CHOP < 38.2 = trend -> breakout continuation). Enter long when price breaks above R1 with 1d uptrend, volume spike, and chop < 38.2 (trend). Enter short when price breaks below S1 with 1d downtrend, volume spike, and chop < 38.2. Uses discrete position size 0.25 to balance capture and drawdown. Designed for 12-30 trades/year on 12h by requiring daily alignment, volume confirmation, and regime filter, reducing overtrading while capturing structured moves in both bull and bear markets. The chop filter avoids whipsaws in ranging markets and improves trend-following accuracy.
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
    
    # Get 1d data for Camarilla levels, trend filter, and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    
    camarilla_range = prev_1d_high - prev_1d_low
    r1 = prev_1d_close + 1.1 * camarilla_range / 12
    s1 = prev_1d_close - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 12h timeframe (no additional delay needed as they're based on completed 1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) over period)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d bars
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.maximum(hl_range, 1e-10)
    
    # Choppiness Index
    chop = 100 * np.log10(atr_14 * 14 / np.log10(hl_range)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA34 warmup, volume MA warmup, chop warmup
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Regime filter: chop < 38.2 = trending (favor breakouts)
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume spike + trending regime
            long_signal = (close[i] > r1_aligned[i]) and trend_1d_uptrend and volume_spike[i] and is_trending
            
            # Short: price breaks below S1 + 1d downtrend + volume spike + trending regime
            short_signal = (close[i] < s1_aligned[i]) and trend_1d_downtrend and volume_spike[i] and is_trending
            
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
            # Exit: price breaks below S1 OR 1d trend turns down OR chop becomes ranging (exit early in ranging)
            if (close[i] < s1_aligned[i] or not trend_1d_uptrend or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR 1d trend turns up OR chop becomes ranging (exit early in ranging)
            if (close[i] > r1_aligned[i] or not trend_1d_downtrend or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0