#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Width regime + 12h Camarilla pivot breakout with volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- Regime filter: Bollinger Band Width percentile from 12h to detect ranging (CHOP) vs trending markets.
- In ranging markets (BW > 60th percentile): fade at Camarilla H3/L3 levels with mean reversion.
- In trending markets (BW < 40th percentile): breakout continuation at Camarilla H4/L4 levels.
- Volume confirmation: 6h volume > 1.5 * 20-period average to ensure participation.
- Signal size: 0.25 discrete to minimize fee drag.
- This adaptive regime approach works in both bull and bear markets by switching between mean reversion and trend following based on market structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def bollinger_band_width(close, period=20, std_dev=2.0):
    """Calculate Bollinger Band Width: (upper - lower) / middle."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / (sma + 1e-10)
    return width, sma, upper, lower

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels for the period."""
    typical_price = (high + low + close) / 3.0
    range_val = high - low
    h4 = close + range_val * 1.1 / 2
    h3 = close + range_val * 1.1 / 4
    l3 = close - range_val * 1.1 / 4
    l4 = close - range_val * 1.1 / 2
    return h4, h3, l3, l4

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Bollinger Band Width for regime detection
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:  # Need sufficient data for BBW
        return np.zeros(n)
    
    bbw_12h, _, _, _ = bollinger_band_width(df_12h['close'].values, 20, 2.0)
    bbw_12h_aligned = align_htf_to_ltf(prices, df_12h, bbw_12h, additional_delay_bars=1)
    
    # Calculate Bollinger Band Width percentile rank (using 50-period lookback)
    bbw_percentile = np.full(len(bbw_12h_aligned), np.nan)
    for i in range(50, len(bbw_12h_aligned)):
        if not np.isnan(bbw_12h_aligned[i]):
            window = bbw_12h_aligned[max(0, i-50):i+1]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) >= 10:
                rank = (np.sum(valid_window <= bbw_12h_aligned[i]) / len(valid_window)) * 100
                bbw_percentile[i] = rank
    
    # Calculate 12h Camarilla pivot levels
    h4_12h, h3_12h, l3_12h, l4_12h = calculate_camarilla_pivots(
        df_12h['high'].values, df_12h['low'].values, df_12h['close'].values
    )
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h, additional_delay_bars=1)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h, additional_delay_bars=1)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h, additional_delay_bars=1)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h, additional_delay_bars=1)
    
    # Calculate 6h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need BBW percentile and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bbw_percentile[i]) or np.isnan(h3_12h_aligned[i]) or 
            np.isnan(l3_12h_aligned[i]) or np.isnan(h4_12h_aligned[i]) or 
            np.isnan(l4_12h_aligned[i]) or i >= len(vol_ma_20) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_bbw_percentile = bbw_percentile[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirmed = curr_volume > 1.5 * vol_ma_20[i]
        
        # Exit conditions
        if position != 0:
            if position == 1:  # Long position
                # Exit long: price crosses below H3 (mean reversion) or L4 (stop)
                if curr_close < h3_12h_aligned[i] or curr_low < l4_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            elif position == -1:  # Short position
                # Exit short: price crosses above L3 (mean reversion) or H4 (stop)
                if curr_close > l3_12h_aligned[i] or curr_high > h4_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions
        if position == 0 and volume_confirmed:
            # Regime detection: ranging vs trending
            if curr_bbw_percentile > 60:  # Ranging market (high volatility, mean revert)
                # Fade at H3/L3 levels
                if curr_close <= h3_12h_aligned[i] and curr_low < h3_12h_aligned[i]:
                    # Potential short at H3 resistance
                    signals[i] = -0.25
                    position = -1
                elif curr_close >= l3_12h_aligned[i] and curr_high > l3_12h_aligned[i]:
                    # Potential long at L3 support
                    signals[i] = 0.25
                    position = 1
            elif curr_bbw_percentile < 40:  # Trending market (low volatility, trend follow)
                # Breakout continuation at H4/L4 levels
                if curr_close > h4_12h_aligned[i] and curr_high > h4_12h_aligned[i]:
                    # Breakout above H4 - go long
                    signals[i] = 0.25
                    position = 1
                elif curr_close < l4_12h_aligned[i] and curr_low < l4_12h_aligned[i]:
                    # Breakdown below L4 - go short
                    signals[i] = -0.25
                    position = -1
        
        # Maintain position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_BBWRegime_CamarillaBreakout_12hPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0