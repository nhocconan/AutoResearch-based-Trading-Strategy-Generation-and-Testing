#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend detection (bull/bear regime).
- Williams %R(14): Oversold < -80 for long, Overbought > -20 for short.
- Trend filter: Only trade mean reversion in alignment with weekly trend (price > EMA50 for longs, price < EMA50 for shorts).
- Volume confirmation: Current volume > 1.5 * 20-period average volume to avoid low-liquidity false signals.
- Exit: Opposite Williams %R level (long exit when %R > -50, short exit when %R < -50).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets by buying dips in uptrend, and in bear markets by selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
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
    
    # Calculate Williams %R(14) on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need sufficient data for Williams %R
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_1d) / (highest_high - lowest_low)) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 1d timeframe (same timeframe, so direct alignment)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 50)  # Need 14 for Williams %R, 20 for volume MA, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r_aligned[i]
        
        # Trend filter: price > EMA50 for uptrend, price < EMA50 for downtrend
        uptrend = curr_close > ema50_1w_aligned[i]
        downtrend = curr_close < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Williams %R levels
        oversold = curr_williams_r < -80
        overbought = curr_williams_r > -20
        exit_long = curr_williams_r > -50
        exit_short = curr_williams_r < -50
        
        # Exit conditions: opposite Williams %R level
        if position != 0:
            # Exit long: Williams %R > -50
            if position == 1:
                if exit_long:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R < -50
            elif position == -1:
                if exit_short:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with trend and volume filters
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume confirmation
            long_condition = oversold and uptrend and volume_confirm
            
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume confirmation
            short_condition = overbought and downtrend and volume_confirm
            
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

name = "1d_WilliamsR_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0