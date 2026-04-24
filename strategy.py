#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend direction and ATR-based volatility filter.
- Donchian(20): Upper/lower bands from 20-period high/low on 1d.
- Trend Filter: Price > EMA50(1w) for long bias, Price < EMA50(1w) for short bias.
- Volume Confirmation: Current volume > 1.5 * 20-period average volume on 1d.
- Entry: Long when close breaks above Donchian upper band AND long bias AND volume confirmation.
         Short when close breaks below Donchian lower band AND short bias AND volume confirmation.
- Exit: Opposite Donchian band touch (long exits when close < lower band, short exits when close > upper band).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 1w trend and filtering breakouts with volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1w ATR(14) for volatility filter (optional regime filter)
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w_arr, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w_arr, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    # Calculate 1d volume average for confirmation (20-period)
    vol_ma_20_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) bands on 1d
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 20)  # Need 20 for Donchian, 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(atr14_1w_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA50(1w) for long bias, price < EMA50(1w) for short bias
        long_bias = curr_close > ema50_1w_aligned[i]
        short_bias = curr_close < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d[i]
        
        # Donchian breakout conditions
        upper_break = curr_high > highest_high[i]  # Break above upper band
        lower_break = curr_low < lowest_low[i]     # Break below lower band
        
        # Exit conditions: opposite Donchian band touch
        if position != 0:
            # Exit long: close below lower Donchian band
            if position == 1:
                if curr_close < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close above upper Donchian band
            elif position == -1:
                if curr_close > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above upper band AND long bias AND volume confirmation
            long_condition = upper_break and long_bias and volume_confirm
            
            # Short: break below lower band AND short bias AND volume confirmation
            short_condition = lower_break and short_bias and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0