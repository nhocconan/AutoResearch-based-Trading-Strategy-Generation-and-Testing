#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend direction and volume spike filter.
- Williams %R(14): Oversold < -80, Overbought > -20.
- Trend Filter: Price > EMA50(1d) for long bias, Price < EMA50(1d) for short bias.
- Volume Confirmation: Current volume > 1.5 * 20-period average volume.
- Entry: Long when %R crosses above -80 AND price > EMA50 AND volume confirmation.
         Short when %R crosses below -20 AND price < EMA50 AND volume confirmation.
- Exit: Opposite %R level (long exits when %R > -20, short exits when %R < -80).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 1d trend and fading extremes only with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Williams %R(14) on 6h timeframe
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, -100 * (highest_high - close) / rr, -50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50)  # Need 14 for %R, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Trend filter: price > EMA50 for long bias, price < EMA50 for short bias
        long_bias = curr_close > ema50_1d_aligned[i]
        short_bias = curr_close < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Williams %R crossovers
        crossed_above_oversold = (prev_williams_r <= -80) and (curr_williams_r > -80)
        crossed_below_overbought = (prev_williams_r >= -20) and (curr_williams_r < -20)
        
        # Exit conditions: opposite %R levels
        if position != 0:
            # Exit long: %R crosses above -20 (overbought territory)
            if position == 1:
                if curr_williams_r > -20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: %R crosses below -80 (oversold territory)
            elif position == -1:
                if curr_williams_r < -80:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R reversal with trend and volume filters
        if position == 0:
            # Long: %R crosses above -80 (oversold) AND long bias AND volume confirmation
            long_condition = crossed_above_oversold and long_bias and volume_confirm
            
            # Short: %R crosses below -20 (overbought) AND short bias AND volume confirmation
            short_condition = crossed_below_overbought and short_bias and volume_confirm
            
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

name = "6h_WilliamsR_Reversal_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0