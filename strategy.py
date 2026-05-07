#!/usr/bin/env python3
# 6H_ChaikinMoneyFlow_1DTrend_RangeFilter
# Hypothesis: 6-hour strategy combining daily Chaikin Money Flow (CMF) for institutional flow confirmation,
# 1-day EMA50 trend filter, and Bollinger Bands width regime filter to avoid low-volatility whipsaws.
# CMF > 0.05 indicates buying pressure, CMF < -0.05 selling pressure.
# Trend filter: price > EMA50 for longs, price < EMA50 for shorts.
# Regime filter: BB width > 20th percentile ensures sufficient volatility for meaningful moves.
# Designed for 15-35 trades/year to minimize fee drag while capturing institutional flow shifts.

name = "6H_ChaikinMoneyFlow_1DTrend_RangeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for CMF calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50 and CMF
        return np.zeros(n)
    
    # Calculate daily Chaikin Money Flow (CMF) over 20 periods
    # CMF = ADV(20) / Volume(20) where ADV = Accumulation Distribution Value
    # ADV multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = df_1d['high'] - df_1d['low']
    hl_range = hl_range.replace(0, np.nan)  # Prevent division by zero
    cmf_multiplier = ((df_1d['close'] - df_1d['low']) - (df_1d['high'] - df_1d['close'])) / hl_range
    cmf_volume = cmf_multiplier * df_1d['volume']
    # ADV(20) = sum of cmf_volume over 20 periods
    adv_20 = cmf_volume.rolling(window=20, min_periods=20).sum()
    vol_20 = df_1d['volume'].rolling(window=20, min_periods=20).sum()
    cmf_20 = adv_20 / vol_20  # Chaikin Money Flow
    cmf_values = cmf_20.fillna(0).values  # Fill NaN with 0 (no flow)
    
    # Calculate 1-day EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Bollinger Bands width (20,2) for regime filter on 1d
    bb_middle = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean()
    bb_std = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_width_percentile = bb_width.rolling(window=50, min_periods=10).rank(pct=True) * 100
    # Use 20th percentile as threshold: only trade when volatility is above low-vol regime
    bb_width_filter = bb_width_percentile >= 20  # True when BB width > 20th percentile
    
    # Align all 1d indicators to 6h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf_values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    bb_width_filter_aligned = align_htf_to_ltf(prices, df_1d, bb_width_filter.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure we have EMA50 and CMF data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(cmf_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(bb_width_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: sufficient volatility (BB width > 20th percentile)
        volatility_filter = bb_width_filter_aligned[i]
        
        if position == 0:
            # Long: CMF > 0.05 (buying pressure) + uptrend (price > EMA50) + volatility filter
            if (cmf_aligned[i] > 0.05 and 
                close[i] > ema_50_aligned[i] and   # Uptrend filter
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.05 (selling pressure) + downtrend (price < EMA50) + volatility filter
            elif (cmf_aligned[i] < -0.05 and 
                  close[i] < ema_50_aligned[i] and   # Downtrend filter
                  volatility_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. CMF crosses back to neutral zone (-0.05 to 0.05) - flow weakening
            # 2. Price crosses EMA50 in opposite direction - trend change
            cmf_neutral = abs(cmf_aligned[i]) <= 0.05
            trend_change = (position == 1 and close[i] < ema_50_aligned[i]) or \
                           (position == -1 and close[i] > ema_50_aligned[i])
            
            if cmf_neutral or trend_change:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals