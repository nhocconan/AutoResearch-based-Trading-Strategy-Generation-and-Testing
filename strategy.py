#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend direction and volume spike filter.
- Donchian Channel: Upper = 20-period high, Lower = 20-period low on 1d.
- Trend Filter: Price > EMA50(1w) for long bias, Price < EMA50(1w) for short bias.
- Volume Confirmation: Current volume > 2.0 * 20-period average volume (strong spike).
- Entry: Long when price crosses above Upper Band AND price > EMA50(1w) AND volume confirmation.
         Short when price crosses below Lower Band AND price < EMA50(1w) AND volume confirmation.
- Exit: Opposite Donchian band touch (long exits when price touches Lower Band, short exits when price touches Upper Band).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 1w trend and requiring strong volume confirmation for breakouts.
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
    
    # Calculate 1w volume average for confirmation (20-period)
    if len(df_1w) < 20:
        return np.zeros(n)
    
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate Donchian Channel on 1d timeframe (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50)  # Need 20 for Donchian, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA50 for long bias, price < EMA50 for short bias
        long_bias = curr_close > ema50_1w_aligned[i]
        short_bias = curr_close < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1w_aligned[i] if not np.isnan(vol_ma_20_1w_aligned[i]) else False
        
        # Donchian bands
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        
        # Breakout conditions
        broke_above_upper = curr_high > upper_band  # Price breaks above upper band
        broke_below_lower = curr_low < lower_band   # Price breaks below lower band
        
        # Exit conditions: touch opposite band
        if position != 0:
            # Exit long: price touches or goes below lower band
            if position == 1:
                if curr_low <= lower_band:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price touches or goes above upper band
            elif position == -1:
                if curr_high >= upper_band:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above upper band AND long bias AND volume confirmation
            long_condition = broke_above_upper and long_bias and volume_confirm
            
            # Short: break below lower band AND short bias AND volume confirmation
            short_condition = broke_below_lower and short_bias and volume_confirm
            
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