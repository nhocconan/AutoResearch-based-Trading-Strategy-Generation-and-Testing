#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 1d EMA50 trend filter and choppiness regime filter to avoid whipsaws. 
Long when price breaks above R1 + 1d uptrend + choppy market (mean reversion favorable). 
Short when price breaks below S1 + 1d downtrend + choppy market. 
Requires volume > 1.3x 24-period average for confirmation. 
Exit on opposite Camarilla level touch or trend reversal. 
Position size: 0.25 to limit drawdown and enable discrete levels. 
Target: 75-200 total trades over 4 years = 19-50/year. 
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets by using 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, trend filter, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Calculate daily EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 24-period average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_24 = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    vol_ma_24_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_24)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range_1d = high_1d - low_1d
    r1_1d = close_1d + (1.1 * hl_range_1d / 12)  # R1 = close + 1.1*(high-low)/12
    s1_1d = close_1d - (1.1 * hl_range_1d / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Align Camarilla levels to match prices index
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate daily choppiness index for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (max(high,n) - min(low,n))) / log10(n)
    # Simplified: use true range and rolling min/max
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.maximum(max_high_14 - min_low_14, 1e-10)
    chop_raw = 100 * np.log10(atr14 * 14 / chop_denom) / np.log10(14)
    chop_1d = np.where(chop_denom > 0, chop_raw, 50.0)  # default to 50 if invalid
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), volume MA (24), ATR14 (14)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_24_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above daily EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 24-period average
        volume_confirm = volume[i] > 1.3 * vol_ma_24_aligned[i]
        
        # Choppiness regime filter: CHOP > 50 indicates choppy/ranging market (favorable for mean reversion near pivots)
        chop_regime = chop_1d_aligned[i] > 50
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 1d uptrend + volume confirmation + choppy regime
            long_setup = (close[i] > r1_aligned[i]) and htf_1d_bullish and volume_confirm and chop_regime
            
            # Short setup: price breaks below Camarilla S1 + 1d downtrend + volume confirmation + choppy regime
            short_setup = (close[i] < s1_aligned[i]) and htf_1d_bearish and volume_confirm and chop_regime
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Camarilla S1 (stop) OR 1d trend turns bearish OR chop regime ends (trending market)
            if (close[i] <= s1_aligned[i]) or (not htf_1d_bullish) or (not chop_regime):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (stop) OR 1d trend turns bullish OR chop regime ends (trending market)
            if (close[i] >= r1_aligned[i]) or (htf_1d_bullish) or (not chop_regime):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0