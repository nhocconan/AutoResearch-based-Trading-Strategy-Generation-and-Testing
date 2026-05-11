#!/usr/bin/env python3
"""
6H_WickReversal_VolumeSpike_1wTrend
Hypothesis: On 6h timeframe, long when price rejects lower wick (bullish engulfing) with volume spike during weekly uptrend;
short when price rejects upper wick (bearish engulfing) with volume spike during weekly downtrend.
Weekly trend filter avoids counter-trend trades. Volume spike confirms institutional interest.
Targets 15-30 trades/year to minimize fee drag in 2025 ranging market.
"""

name = "6H_WickReversal_VolumeSpike_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA21 for trend
    ema21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Weekly trend: above/below EMA21
    weekly_uptrend = ema21_1w_aligned > 0  # Will be refined below
    weekly_downtrend = ema21_1w_aligned > 0  # Placeholder, will compute properly
    
    # Actually compute trend direction properly
    weekly_uptrend = close_1w > ema21_1w
    weekly_downtrend = close_1w < ema21_1w
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Volume spike filter: 24-period EMA (approx 6 days on 6h)
    vol_ema24 = pd.Series(volume).ewm(span=24, min_periods=24, adjust=False).mean().values
    volume_ok = volume > vol_ema24 * 2.0  # Require 2x average volume
    
    # Wick rejection detection: bullish/bearish engulfing
    # Bullish: current close > prior open AND current open < prior close (engulfing)
    # Bearish: current close < prior open AND current open > prior close (engulfing)
    bullish_engulf = (close > np.roll(open_price, 1)) & (open_price < np.roll(close, 1))
    bearish_engulf = (close < np.roll(open_price, 1)) & (open_price > np.roll(close, 1))
    
    # Handle first element
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    # Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        is_bullish_engulf = bullish_engulf[i]
        is_bearish_engulf = bearish_engulf[i]
        vol_spike = volume_ok[i]
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Long: Bullish engulfing + volume spike + weekly uptrend
            if is_bullish_engulf and vol_spike and weekly_up:
                signals[i] = position_size
                position = 1
            # Short: Bearish engulfing + volume spike + weekly downtrend
            elif is_bearish_engulf and vol_spike and weekly_down:
                signals[i] = -position_size
                position = -1
        else:
            # Exit on opposite engulfing signal or trend change
            if position == 1:
                # Exit: Bearish engulfing OR weekly trend turns down
                if is_bearish_engulf or weekly_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Bullish engulfing OR weekly trend turns up
                if is_bullish_engulf or weekly_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals