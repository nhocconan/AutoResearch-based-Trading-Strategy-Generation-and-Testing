#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and 1d volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume spike confirmation and 1w for pivot-based trend direction.
- Donchian breakout: Long when price > highest high of last 20 periods, Short when price < lowest low of last 20 periods.
- Weekly pivot direction: Use 1w Camarilla H3/L3 levels to determine bias - only take longs above H3, shorts below L3.
- Volume confirmation: Require 1d volume > 2.0 * 20-period average volume to ensure participation.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Weekly pivot filter prevents counter-trend trades in ranging markets.
- Volume confirmation reduces false breakouts.
- Designed to work in both bull (breakouts with volume) and bear (breakdowns with volume) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d, additional_delay_bars=1)
    
    # Calculate 1w Camarilla pivot levels for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Camarilla pivot calculation: based on previous week's OHLC
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # H3 and L3 are the key levels for intraday trading
    prev_week_high = df_1w['high'].shift(1).values  # Previous week high
    prev_week_low = df_1w['low'].shift(1).values    # Previous week low
    prev_week_close = df_1w['close'].shift(1).values # Previous week close
    
    # Calculate Camarilla H3 and L3 levels
    diff = prev_week_high - prev_week_low
    camarilla_h3 = prev_week_close + 1.125 * diff
    camarilla_l3 = prev_week_close - 1.125 * diff
    
    # Align weekly levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3, additional_delay_bars=1)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3, additional_delay_bars=1)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need sufficient data for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol_ratio = vol_ratio_1d_aligned[i]  # Use 1d volume ratio
        
        # Exit conditions: opposite Donchian breakout OR loss of volume confirmation
        if position != 0:
            # Exit long: price breaks below lowest low OR volume drops below threshold
            if position == 1:
                if curr_low < lowest_low[i] or curr_vol_ratio < 1.5:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above highest high OR volume drops below threshold
            elif position == -1:
                if curr_high > highest_high[i] or curr_vol_ratio < 1.5:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with pivot direction filter and volume confirmation
        if position == 0:
            # Long: price breaks above highest high AND above weekly H3 AND volume confirmation
            if (curr_high > highest_high[i] and 
                curr_close > camarilla_h3_aligned[i] and 
                curr_vol_ratio > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lowest low AND below weekly L3 AND volume confirmation
            elif (curr_low < lowest_low[i] and 
                  curr_close < camarilla_l3_aligned[i] and 
                  curr_vol_ratio > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1wCamarilla_PivotDirection_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0