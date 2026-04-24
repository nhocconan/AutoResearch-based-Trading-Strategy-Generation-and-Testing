#!/usr/bin/env python3
"""
6h Williams %R + 1d EMA34 Trend + Volume Spike
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and volume spike confirmation.
- Williams %R(14): Oversold < -80 for long, overbought > -20 for short.
- Trend: Price > 1d EMA34 for longs, Price < 1d EMA34 for shorts.
- Volume Spike: Current 6h volume > 2.5 * 20-period 1d average volume (aligned).
- Exit: Opposite Williams %R signal (long exit when %R > -50, short exit when %R < -50).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets via trend-following longs, in bear via trend-following shorts.
- Avoids whipsaws by requiring both momentum extreme and trend alignment.
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
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 6h Williams %R(14)
    williams_window = 14
    highest_high = pd.Series(high).rolling(window=williams_window, min_periods=williams_window).max().values
    lowest_low = pd.Series(low).rolling(window=williams_window, min_periods=williams_window).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    numerator = highest_high - close
    denominator = highest_high - lowest_low
    # Avoid division by zero
    williams_r = np.where(denominator != 0, (numerator / denominator) * -100, -50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(williams_window, 34)  # Need 14 for Williams %R, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema34_1d_aligned[i]
        downtrend = curr_close < ema34_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.5 * 20-period average volume
        volume_confirm = curr_volume > 2.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Williams %R signal
        if position != 0:
            # Exit long: Williams %R > -50 (moving out of oversold)
            if position == 1:
                if curr_williams_r > -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R < -50 (moving out of overbought)
            elif position == -1:
                if curr_williams_r < -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with trend and volume filters
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume confirmation
            long_condition = (curr_williams_r < -80 and 
                            uptrend and
                            volume_confirm)
            
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume confirmation
            short_condition = (curr_williams_r > -20 and 
                             downtrend and
                             volume_confirm)
            
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

name = "6h_WilliamsR_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0