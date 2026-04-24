#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d HMA(21) trend filter + volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for HMA21 trend filter to capture major trend direction on higher timeframe.
- Donchian(20): Price breaking above/below 20-period high/low on 12h chart indicates momentum shift.
- Entry: Long when price > Donchian Upper(20) AND price > 1d HMA21 AND volume > 1.5 * 20-period average volume.
         Short when price < Donchian Lower(20) AND price < 1d HMA21 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian break OR price crosses 1d HMA21 in opposite direction.
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Donchian breakouts capture strong moves; 1d HMA21 filter ensures alignment with major trend.
- Volume confirmation reduces false breakouts. Works in both bull (breakouts continuation) and bear (breakdown continuation).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def hma(values, period):
    """Calculate Hull Moving Average with proper min_periods."""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(values).ewm(span=half, adjust=False, min_periods=half).mean()
    wma1 = pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean()
    raw = 2 * wma2 - wma1
    hma_val = pd.Series(raw).ewm(span=sqrt, adjust=False, min_periods=sqrt).mean()
    return hma_val.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for 12h indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    if n < 20:
        return np.zeros(n)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d trend filter: HMA21
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    hma21_1d = hma(df_1d['close'].values, 21)
    hma21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma21_1d)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 60)  # Need Donchian(20) and 1d HMA21
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma21_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian break OR price crosses 1d HMA21 in opposite direction
        if position != 0:
            # Exit long: price < Donchian Lower(20) OR price falls below 1d HMA21
            if position == 1:
                if curr_close < lowest_low[i] or curr_close < hma21_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > Donchian Upper(20) OR price rises above 1d HMA21
            elif position == -1:
                if curr_close > highest_high[i] or curr_close > hma21_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian break with trend filter and volume confirmation
        if position == 0:
            # Long: Price > Donchian Upper(20) AND price > 1d HMA21 AND volume confirmation
            long_condition = (curr_close > highest_high[i] and 
                            curr_close > hma21_1d_aligned[i] and
                            curr_volume > 1.5 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False)
            
            # Short: Price < Donchian Lower(20) AND price < 1d HMA21 AND volume confirmation
            short_condition = (curr_close < lowest_low[i] and 
                             curr_close < hma21_1d_aligned[i] and
                             curr_volume > 1.5 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False)
            
            if long_condition:
                signals[i] = 0.30
                position = 1
            elif short_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_Breakout_1dHMA21_TrendFilter_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0