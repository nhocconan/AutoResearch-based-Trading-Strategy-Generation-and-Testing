#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and 1w EMA50 trend filter
- Donchian channel breakouts capture momentum moves with clear structure
- 1d ATR(14) filter ensures volatility is sufficient (ATR > 0.5 * 20-period ATR MA) to avoid chop
- 1w EMA(50) trend filter ensures we only trade breakouts in the weekly trend direction
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by aligning with weekly trend
- Volume confirmation (> 1.5x 20-period average) ensures breakout has participation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar has no previous close
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_filter = atr_14 > (0.5 * atr_ma_20)  # Volatility sufficient if ATR > 50% of its MA
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or
            i >= len(atr_filter) or np.isnan(atr_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        # Long: price breaks above Donchian high with volume and filters
        # Short: price breaks below Donchian low with volume and filters
        price_above_dc_high = close[i] > donchian_high[i]
        price_below_dc_low = close[i] < donchian_low[i]
        
        # Trend filter: weekly EMA50 direction
        # Need previous weekly EMA to determine slope
        if i > 0 and not np.isnan(ema_50_1w_aligned[i-1]):
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            weekly_uptrend = True  # Default to allow trading until we have slope
            weekly_downtrend = True
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, weekly uptrend, volume spike, sufficient volatility
            long_signal = (price_above_dc_high and 
                          weekly_uptrend and
                          volume[i] > 1.5 * vol_ma[i] and
                          atr_filter[i])
            
            # Short conditions: price breaks below Donchian low, weekly downtrend, volume spike, sufficient volatility
            short_signal = (price_below_dc_low and 
                           weekly_downtrend and
                           volume[i] > 1.5 * vol_ma[i] and
                           atr_filter[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian break or weekly trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below Donchian low or weekly trend turns down
                if (price_below_dc_low or 
                    not weekly_uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above Donchian high or weekly trend turns up
                if (price_above_dc_high or 
                    not weekly_downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_1dATRFilter_1wEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0