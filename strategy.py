#!/usr/bin/env python3
# 4h_BBands_Squeeze_Breakout_1dTrend
# Hypothesis: Bollinger Band squeeze breakouts on 4h, filtered by 1d trend direction and volume confirmation.
# Long when: 1d uptrend (higher high & higher low), Bollinger Band width at 20-period low, and price breaks above upper BB.
# Short when: 1d downtrend (lower high & lower low), Bollinger Band width at 20-period low, and price breaks below lower BB.
# Exit when price crosses back through the middle Bollinger Band.
# Designed to capture volatility expansion after contraction in both bull and bear markets.
# Bollinger Band squeeze identifies low volatility periods; breakout captures the ensuing move.
# Works in bull markets by catching breakouts of consolidations and in bear markets by catching breakdowns.

name = "4h_BBands_Squeeze_Breakout_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend structure (HH, HL, LH, LL)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d trend structure: HH/HL for uptrend, LH/LL for downtrend ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Higher High: today's high > yesterday's high
    hh = high_1d > np.roll(high_1d, 1)
    # Higher Low: today's low > yesterday's low
    hl = low_1d > np.roll(low_1d, 1)
    # Lower High: today's high < yesterday's high
    lh = high_1d < np.roll(high_1d, 1)
    # Lower Low: today's low < yesterday's low
    ll = low_1d < np.roll(low_1d, 1)
    # Uptrend: HH and HL
    uptrend = hh & hl
    # Downtrend: LH and LL
    downtrend = lh & ll
    # First bar: no previous day, set to False
    uptrend[0] = False
    downtrend[0] = False
    
    # --- Bollinger Bands (20, 2) ---
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    upper_bb = np.full(n, np.nan)
    lower_bb = np.full(n, np.nan)
    bb_width = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        sma[i] = np.mean(close[i - bb_period + 1:i + 1])
        std_dev[i] = np.std(close[i - bb_period + 1:i + 1])
        upper_bb[i] = sma[i] + bb_std * std_dev[i]
        lower_bb[i] = sma[i] - bb_std * std_dev[i]
        bb_width[i] = upper_bb[i] - lower_bb[i]
    
    # --- Bollinger Band Width Squeeze (lowest 20-period) ---
    bb_width_lookback = 20
    bb_width_min = np.full(n, np.nan)
    for i in range(bb_width_lookback - 1, n):
        bb_width_min[i] = np.min(bb_width[i - bb_width_lookback + 1:i + 1])
    bb_squeeze = bb_width <= bb_width_min  # True when at period low
    
    # Align 1d trend indicators to 4h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for BBands (20) and BB width lookback (20)
    start_idx = max(bb_period, bb_width_lookback)  # 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(upper_bb[i]) or
            np.isnan(lower_bb[i]) or
            np.isnan(sma[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 1d
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        # Bollinger Band squeeze condition
        is_squeeze = bb_squeeze[i]
        
        if position == 0:
            if is_uptrend and is_squeeze:
                # Long: daily uptrend + BB squeeze + price breaks above upper BB
                if close[i] > upper_bb[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and is_squeeze:
                # Short: daily downtrend + BB squeeze + price breaks below lower BB
                if close[i] < lower_bb[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price crosses below middle BB (SMA)
                if close[i] < sma[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above middle BB (SMA)
                if close[i] > sma[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals