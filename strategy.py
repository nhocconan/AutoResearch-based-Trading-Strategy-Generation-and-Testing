#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width regime filter + 1d ADX trend confirmation
# Bollinger Band Width (BBW) identifies ranging vs trending markets
# Low BBW (< 20th percentile) = squeeze → potential breakout
# High BBW (> 80th percentile) = expansion → trend continuation
# 1d ADX > 25 confirms trend strength on higher timeframe
# Entry long: BBW < 20th percentile AND price > upper BBAND AND 1d ADX > 25
# Entry short: BBW < 20th percentile AND price < lower BBAND AND 1d ADX > 25
# Exit: When BBW > 50th percentile (squeeze resolved) OR opposite BBAND touch
# Uses volatility contraction/expansion for timing, HTF ADX for trend validation
# Timeframe: 6h, HTF: 1d. Target: 80-180 total trades over 4 years (20-45/year)

name = "6h_BBW_1dADX_SqueezeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(close_1d) >= 14:
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / np.where(atr_14 == 0, np.nan, atr_14)
        di_minus = 100 * dm_minus_smooth / np.where(atr_14 == 0, np.nan, atr_14)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, np.nan, (di_plus + di_minus))
        adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        adx_1d = np.full(len(close_1d), np.nan)
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Bollinger Bands (20, 2)
    if len(close) >= 20:
        ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = ma_20 + (2.0 * std_20)
        lower_bb = ma_20 - (2.0 * std_20)
        bb_width = (upper_bb - lower_bb) / ma_20  # Normalized width
    else:
        ma_20 = np.full(n, np.nan)
        upper_bb = np.full(n, np.nan)
        lower_bb = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # Calculate percentiles of BBW for regime detection (using expanding window)
    bbw_pct_20 = np.full(n, np.nan)
    bbw_pct_50 = np.full(n, np.nan)
    bbw_pct_80 = np.full(n, np.nan)
    
    for i in range(20, n):
        if np.isnan(bb_width[i]):
            continue
        # Use expanding window up to current point for percentile calculation
        hist_bbw = bb_width[20:i+1]  # From BBW warmup to current
        if len(hist_bbw) >= 10:  # Minimum samples for percentile
            bbw_pct_20[i] = np.percentile(hist_bbw, 20)
            bbw_pct_50[i] = np.percentile(hist_bbw, 50)
            bbw_pct_80[i] = np.percentile(hist_bbw, 80)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bbw_pct_20[i]) or np.isnan(bbw_pct_50[i]) or
            np.isnan(ma_20[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for Bollinger Band squeeze (low volatility) breakout
            # BBW below 20th percentile indicates squeeze
            if bb_width[i] < bbw_pct_20[i]:
                # Long breakout: price breaks above upper BBAND
                if close[i] > upper_bb[i] and adx_1d_aligned[i] > 25:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price breaks below lower BBAND
                elif close[i] < lower_bb[i] and adx_1d_aligned[i] > 25:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: BBW expands beyond 50th percentile (squeeze resolved) 
            # OR price touches lower BBAND (mean reversion)
            if bb_width[i] > bbw_pct_50[i] or close[i] < lower_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: BBW expands beyond 50th percentile OR price touches upper BBAND
            if bb_width[i] > bbw_pct_50[i] or close[i] > upper_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals