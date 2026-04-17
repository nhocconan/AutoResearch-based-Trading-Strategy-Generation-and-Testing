#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1w Williams %R extreme + 1d volume spike + 6h price action filter.
Long when 1w Williams %R < -80 (oversold) and 6h close > 6h open (bullish candle) and 1d volume > 2.0x 20-period average.
Short when 1w Williams %R > -20 (overbought) and 6h close < 6h open (bearish candle) and 1d volume > 2.0x 20-period average.
Williams %R identifies exhaustion points on weekly scale; volume spike confirms institutional participation; 6h candle direction filters for immediate price action alignment.
Designed to work in ranging markets (mean reversion from extremes) and trending markets (pullbacks to weekly extreme in trend direction).
Target: 80-180 total trades over 4 years (20-45/year) with discrete sizing 0.25 to minimize fee drag.
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
    open_price = prices['open'].values
    
    # Get 1w data for Williams %R
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w Williams %R (14-period)
    def williams_r(high_vals, low_vals, close_vals, window):
        highest_high = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close_vals) / (highest_high - lowest_low)
        return wr
    
    wr_14_1w = williams_r(high_1w, low_1w, close_1w, 14)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (6h)
    wr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, wr_14_1w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # need enough for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_14_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-day average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_1d_aligned[i]
        # 6h candle direction: bullish (close > open) or bearish (close < open)
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        # Williams %R extremes: oversold (< -80) or overbought (> -20)
        oversold = wr_14_1w_aligned[i] < -80
        overbought = wr_14_1w_aligned[i] > -20
        
        if position == 0:
            # Long: weekly oversold + bullish 6h candle + volume spike
            if oversold and bullish_candle and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: weekly overbought + bearish 6h candle + volume spike
            elif overbought and bearish_candle and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly Williams %R returns above -50 (momentum shift)
            if wr_14_1w_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly Williams %R returns below -50 (momentum shift)
            if wr_14_1w_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wWilliamsR14_VolumeSpike_6hCandleDir"
timeframe = "6h"
leverage = 1.0