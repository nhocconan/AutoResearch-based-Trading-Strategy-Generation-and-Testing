#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm_v1
Hypothesis: Elder Ray Bull/Bear Power combined with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar average) captures strong directional moves on 6h timeframe. Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 and rising + price > 1d EMA50 + volume confirm. Short when Bear Power < 0 and falling + price < 1d EMA50 + volume confirm. Uses discrete sizing (0.25) to target 12-30 trades/year. Works in bull/bear by only taking signals aligned with 1d trend. Volatility-adjusted exit when power weakens.
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # EMA13 for Elder Ray calculation (on 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Slope of Bull/Bear Power (1-bar change)
    bull_power_slope = np.diff(bull_power, prepend=bull_power[0])
    bear_power_slope = np.diff(bear_power, prepend=bear_power[0])
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of EMA13 (13), EMA50 (50), volume MA (20)
    start_idx = max(13, 50, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        bull_slope = bull_power_slope[i]
        bear_slope = bear_power_slope[i]
        trend_val = ema50_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(bull_val) or np.isnan(bear_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1d EMA50 = uptrend, price < 1d EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Entry conditions: Elder Ray aligned with 1d trend + volume
        long_entry = (bull_val > 0) and (bull_slope > 0) and is_uptrend and vol_conf
        short_entry = (bear_val < 0) and (bear_slope < 0) and is_downtrend and vol_conf
        
        # Exit conditions: Elder Ray weakening or opposite signal
        long_exit = False
        short_exit = False
        if position == 1:
            # Long exit: Bull Power turns negative or slope negative
            long_exit = (bull_val <= 0) or (bull_slope <= 0)
        elif position == -1:
            # Short exit: Bear Power turns positive or slope positive
            short_exit = (bear_val >= 0) or (bear_slope >= 0)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0