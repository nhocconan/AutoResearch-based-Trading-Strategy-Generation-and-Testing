#!/usr/bin/env python3
"""
6h_1w_HeikinAshi_Trend_Momentum
Hypothesis: Use Heikin-Ashi candles from weekly timeframe to determine major trend,
and Heikin-Ashi from 6h for entry timing with momentum confirmation.
- Long when: Weekly HA close > Weekly HA open (uptrend) AND 6h HA close > 6h HA open AND 6h momentum > 0
- Short when: Weekly HA close < Weekly HA open (downtrend) AND 6h HA close < 6h HA open AND 6h momentum < 0
- Exit when trend or momentum conditions fail
Heikin-Ashi smooths price action to filter noise and identify true trends.
Combining weekly trend with 6h momentum captures major moves while avoiding whipsaws.
Targets 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
"""

name = "6h_1w_HeikinAshi_Trend_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ha(open_arr, high_arr, low_arr, close_arr):
    """Calculate Heikin-Ashi values"""
    ha_close = (open_arr + high_arr + low_arr + close_arr) / 4
    ha_open = np.zeros_like(ha_close)
    ha_open[0] = (open_arr[0] + close_arr[0]) / 2
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum.reduce([high_arr, ha_open, ha_close])
    ha_low = np.minimum.reduce([low_arr, ha_open, ha_close])
    return ha_open, ha_high, ha_low, ha_close

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend determination
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Weekly OHLC for HA calculation
    open_1w = df_1w['open'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Heikin-Ashi for weekly trend
    ha_open_1w, ha_high_1w, ha_low_1w, ha_close_1w = calculate_ha(
        open_1w, high_1w, low_1w, close_1w
    )
    
    # Calculate Heikin-Ashi for 6h entry signals
    ha_open_6h, ha_high_6h, ha_low_6h, ha_close_6h = calculate_ha(
        open_6h, high_6h, low_6h, close_6h
    )
    
    # 6h momentum: rate of change of HA close
    mom_period = 6
    mom_6h = np.full_like(ha_close_6h, np.nan)
    for i in range(mom_period, len(ha_close_6h)):
        mom_6h[i] = (ha_close_6h[i] - ha_close_6h[i - mom_period]) / ha_close_6h[i - mom_period] * 100.0
    
    # Weekly trend: HA close vs HA open
    weekly_uptrend = ha_close_1w > ha_open_1w
    weekly_downtrend = ha_close_1w < ha_open_1w
    
    # Align weekly trend to 6h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(30, mom_period)  # for HA and momentum
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(ha_open_6h[i]) or np.isnan(ha_close_6h[i]) or np.isnan(mom_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine conditions
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        ha_bullish = ha_close_6h[i] > ha_open_6h[i]
        ha_bearish = ha_close_6h[i] < ha_open_6h[i]
        mom_pos = mom_6h[i] > 0
        mom_neg = mom_6h[i] < 0
        
        if position == 0:
            # Look for entries
            if weekly_up and ha_bullish and mom_pos:
                # Long: weekly uptrend + 6h bullish HA + positive momentum
                signals[i] = 0.25
                position = 1
            elif weekly_down and ha_bearish and mom_neg:
                # Short: weekly downtrend + 6h bearish HA + negative momentum
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: weekly trend turns down OR HA turns bearish OR momentum turns negative
                if not weekly_up or not ha_bullish or not mom_pos:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly trend turns up OR HA turns bullish OR momentum turns positive
                if not weekly_down or not ha_bearish or not mom_neg:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals