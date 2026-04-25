#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrendFilter_VolumeConfirm
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with 1-week EMA34 trend filter and volume confirmation captures institutional buying/selling pressure while avoiding counter-trend trades. Works in bull markets (buy on Bull Power > 0 + uptrend) and bear markets (sell on Bear Power > 0 + downtrend). Target: 12-30 trades/year (50-120 over 4 years) to minimize fee drag. Uses discrete position sizing (0.0, ±0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA34 trend filter
    ema_34_1w = calculate_ema(df_1w['close'].values, 34)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13 = calculate_ema(close, 13)
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying pressure
    bear_power = ema_13 - low   # Selling pressure
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA13 (13) + volume MA (20) + 1w EMA34 (34)
    start_idx = max(13, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:
            # Look for entry signals - require: Elder Ray signal + 1w trend alignment + volume confirmation
            long_entry = (curr_bull_power > 0) and (curr_close > ema_34_1w_aligned[i]) and curr_volume_confirm
            short_entry = (curr_bear_power > 0) and (curr_close < ema_34_1w_aligned[i]) and curr_volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when Bull Power turns negative OR trend turns bearish
            if curr_bull_power <= 0 or curr_close < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Bear Power turns negative OR trend turns bullish
            if curr_bear_power <= 0 or curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1wTrendFilter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0