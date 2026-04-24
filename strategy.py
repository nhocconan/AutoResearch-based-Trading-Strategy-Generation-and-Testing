#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and volume spike detection.
- Williams %R(14): Oversold < -80, Overbought > -20.
- Trend Filter: 1d EMA34 - price > EMA34 = uptrend, < EMA34 = downtrend.
- Volume Confirmation: 6h volume > 1.5 * 20-period average volume.
- Entry: Long when Williams %R crosses above -80 AND uptrend AND volume confirmation.
         Short when Williams %R crosses below -20 AND downtrend AND volume confirmation.
- Exit: Opposite Williams %R level (long exit at > -20, short exit at < -80).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading mean reversions in the direction of the 1d trend.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 6h Williams %R(14)
    williams_window = 14
    highest_high = pd.Series(high).rolling(window=williams_window, min_periods=williams_window).max().values
    lowest_low = pd.Series(low).rolling(window=williams_window, min_periods=williams_window).min().values
    
    # Avoid division by zero
    denom = highest_high - lowest_low
    denom = np.where(denom == 0, 1e-10, denom)
    williams_r = -100 * (highest_high - close) / denom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(williams_window, 34)  # Need 14 for Williams %R, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams = williams_r[i]
        prev_williams = williams_r[i-1] if i > 0 else -50
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema34_aligned[i]
        downtrend = curr_close < ema34_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Williams %R levels
        oversold = -80
        overbought = -20
        
        # Exit conditions: opposite Williams %R level
        if position != 0:
            # Exit long: Williams %R > overbought (-20)
            if position == 1:
                if curr_williams > overbought:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R < oversold (-80)
            elif position == -1:
                if curr_williams < oversold:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R crossover with trend and volume filters
        if position == 0:
            # Long: Williams %R crosses above oversold (-80) AND uptrend AND volume confirmation
            long_condition = (prev_williams <= oversold and curr_williams > oversold and
                            uptrend and volume_confirm)
            
            # Short: Williams %R crosses below overbought (-20) AND downtrend AND volume confirmation
            short_condition = (prev_williams >= overbought and curr_williams < overbought and
                             downtrend and volume_confirm)
            
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

name = "6h_WilliamsR_1dEMA34Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0