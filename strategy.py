#!/usr/bin/env python3
"""
6h Elder Ray + Weekly EMA Trend + Volume Spike
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
In bull markets, buy when Bull Power > 0 and rising; in bear markets, sell when Bear Power > 0 and rising.
Uses 1w EMA34 for primary trend filter and volume confirmation to avoid whipsaws. Targets 50-150 trades over 4 years.
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
    
    # Get 1w data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close (only needs completed 1w candle)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 13-period EMA for Elder Ray on 6h
    ema_13_6h = np.full(n, np.nan)
    for i in range(13, n):
        ema_13_6h[i] = np.mean(close[i-12:i+1]) * (2/(13+1)) + ema_13_6h[i-1] * (1 - 2/(13+1)) if not np.isnan(ema_13_6h[i-1]) else np.mean(close[i-12:i+1])
    # Initialize first value
    if np.isnan(ema_13_6h[12]):
        ema_13_6h[12] = np.mean(close[0:13])
    
    # Calculate Elder Ray components
    bull_power = high - ema_13_6h  # High - EMA13
    bear_power = ema_13_6h - low   # EMA13 - Low
    
    # Calculate 20-period volume MA for 6h volume spike
    vol_ma_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_6h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA13 and volume MA
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma_20_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        weekly_trend = ema_34_1w_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        vol_ma_6h = vol_ma_20_6h[i]
        
        # Volume confirmation: current 6h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_6h
        
        # Elder Ray signals with trend filter
        # Long: Bull Power > 0 (bulls in control) AND rising AND above weekly EMA AND volume confirmation
        long_entry = (curr_bull_power > 0 and 
                     i > start_idx and curr_bull_power > bull_power[i-1] and  # rising
                     curr_close > weekly_trend and 
                     volume_confirm)
        # Short: Bear Power > 0 (bears in control) AND rising AND below weekly EMA AND volume confirmation
        short_entry = (curr_bear_power > 0 and 
                      i > start_idx and curr_bear_power > bear_power[i-1] and  # rising
                      curr_close < weekly_trend and 
                      volume_confirm)
        
        if position == 0:
            # Look for entry signals
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Bull Power <= 0 (bulls lose control) OR below weekly EMA
            if curr_bull_power <= 0 or curr_close < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Bear Power <= 0 (bears lose control) OR above weekly EMA
            if curr_bear_power <= 0 or curr_close > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0