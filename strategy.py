#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend detection and volume spike filter.
- Camarilla levels: H3/L3 from prior day's range (standard formula).
- Trend filter: Price > 1w EMA50 for longs, Price < 1w EMA50 for shorts.
- Volume confirmation: Current volume > 2.0 * 1w average volume.
- Exit: Opposite Camarilla breakout (price < H3 for long exit, price > L3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading breakouts in the direction of the weekly trend.
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
    volume_1w = df_1w['volume'].values
    
    # EMA50 calculation
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1w volume average for confirmation
    vol_ma_1w = pd.Series(volume_1w).rolling(window=50, min_periods=50).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Calculate Camarilla H3/L3 levels from prior day
    # Need prior day's high, low, close
    prior_high = np.concatenate([[np.nan], high[:-1]])
    prior_low = np.concatenate([[np.nan], low[:-1]])
    prior_close = np.concatenate([[np.nan], close[:-1]])
    
    # Camarilla formula: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = prior_close + 1.1 * (prior_high - prior_low) / 4
    camarilla_l3 = prior_close - 1.1 * (prior_high - prior_low) / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need 50 for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend = curr_close > ema50_1w_aligned[i]
        downtrend = curr_close < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 1w average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_1w_aligned[i] if not np.isnan(vol_ma_1w_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price < H3
            if position == 1:
                if curr_close < camarilla_h3[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > L3
            elif position == -1:
                if curr_close > camarilla_l3[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: price > H3 AND uptrend AND volume confirmation
            long_condition = (curr_close > camarilla_h3[i] and 
                            uptrend and
                            volume_confirm)
            
            # Short: price < L3 AND downtrend AND volume confirmation
            short_condition = (curr_close < camarilla_l3[i] and 
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

name = "1d_Camarilla_H3L3_Breakout_1wEMA50Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0