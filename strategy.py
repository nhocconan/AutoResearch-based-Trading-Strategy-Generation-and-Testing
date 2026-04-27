#!/usr/bin/env python3
"""
4h_RangeBreakout_1dTrend_VolumeFilter
Hypothesis: In ranging markets, price often oscillates between clear support/resistance levels. 
We identify ranges using 1d Bollinger Bands (width < 30th percentile) and trade breakouts 
from the range with 1d EMA50 trend filter and volume confirmation. This captures 
institutional breakouts while avoiding false signals in chop. Targets 20-30 trades/year.
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
    
    # Get 1d data for range detection and filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands for range detection
    close_1d = df_1d['close'].values
    bb_mid = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Range filter: Bollinger Band width below 30th percentile indicates ranging market
    # Calculate rolling 50-period percentile of BB width
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.30).values
    is_ranging = bb_width < bb_width_percentile
    
    # Align range filter to 4h
    is_ranging_aligned = align_htf_to_ltf(prices, df_1d, is_ranging.astype(float))
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Range boundaries from previous day's Bollinger Bands
    range_upper = bb_upper  # Upper Bollinger Band as resistance
    range_lower = bb_lower  # Lower Bollinger Band as support
    range_upper_aligned = align_htf_to_ltf(prices, df_1d, range_upper)
    range_lower_aligned = align_htf_to_ltf(prices, df_1d, range_lower)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for BB, EMA, and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(range_upper_aligned[i]) or 
            np.isnan(range_lower_aligned[i]) or
            np.isnan(is_ranging_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        range_upper_level = range_upper_aligned[i]
        range_lower_level = range_lower_aligned[i]
        is_range = is_ranging_aligned[i] > 0.5  # True if ranging
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Only trade in ranging markets
            if is_range:
                # Long: break above range upper with uptrend and volume spike
                if close[i] > range_upper_level and vol_spike_val and close[i] > ema_trend:
                    signals[i] = size
                    position = 1
                # Short: break below range lower with downtrend and volume spike
                elif close[i] < range_lower_level and vol_spike_val and close[i] < ema_trend:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: break below range lower or trend turns down
            if close[i] < range_lower_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: break above range upper or trend turns up
            if close[i] > range_upper_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RangeBreakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0