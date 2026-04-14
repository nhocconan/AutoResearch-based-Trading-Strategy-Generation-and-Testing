#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band Squeeze Breakout with 1d Volume Confirmation and ADX Trend Filter
# Takes long when price breaks above upper Bollinger Band during low volatility (squeeze) with 1d volume spike and ADX > 25
# Takes short when price breaks below lower Bollinger Band during low volatility (squeeze) with 1d volume spike and ADX > 25
# Bollinger Band squeeze defined as bandwidth < 20th percentile of last 50 periods
# Exits when price crosses back inside Bollinger Bands or volatility expands (bandwidth > 50th percentile)
# Designed to capture explosive moves after consolidation periods, avoiding choppy markets
# Target: 15-40 trades per symbol over 4 years (4-10/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h Bollinger Bands (20-period, 2 std)
    close_12h = df_12h['close'].values
    bb_middle = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized bandwidth
    
    # Calculate Bollinger Band width percentiles for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_lower_thresh = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    bb_width_upper_thresh = bb_width_series.rolling(window=50, min_periods=20).quantile(0.50).values
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_12h, bb_middle)
    bb_width_lower_thresh_aligned = align_htf_to_ltf(prices, df_12h, bb_width_lower_thresh)
    bb_width_upper_thresh_aligned = align_htf_to_ltf(prices, df_12h, bb_width_upper_thresh)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for Bollinger Bands and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_width_lower_thresh_aligned[i]) or np.isnan(bb_width_upper_thresh_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_width_current = bb_width[i] if i < len(bb_width) else bb_width[-1]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Look for Bollinger Band squeeze breakout with volume and trend confirmation
            is_squeeze = bb_width_current < bb_width_lower_thresh_aligned[i]
            if is_squeeze:
                # Long setup: break above upper BB with volume spike and strong trend
                if (price > bb_upper_aligned[i] and 
                    vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                    adx_aligned[i] > 25):                           # Strong trend
                    position = 1
                    signals[i] = position_size
                # Short setup: break below lower BB with volume spike and strong trend
                elif (price < bb_lower_aligned[i] and 
                      vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                      adx_aligned[i] > 25):                           # Strong trend
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below middle BB or volatility expands (end of squeeze)
            if price < bb_middle_aligned[i] or bb_width_current > bb_width_upper_thresh_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above middle BB or volatility expands (end of squeeze)
            if price > bb_middle_aligned[i] or bb_width_current > bb_width_upper_thresh_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Bollinger_Squeeze_Breakout_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0