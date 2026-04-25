#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeSpike
Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 1-day EMA34 trend filter and volume confirmation.
Targets 12-30 trades/year by requiring: 1) Bull/Bear Power crosses zero line with 1d trend alignment,
2) volume > 1.8x 20-period average. Uses 6h timeframe to reduce fee drag while capturing intermediate swings.
Elder Ray measures buying/selling pressure relative to EMA13; zero crosses indicate momentum shifts.
Works in bull markets via long signals on Bull Power crosses, in bear markets via short signals on Bear Power crosses.
Volume spike filter ensures participation, reducing false signals in low-activity periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA34 trend filter and EMA13 for Elder Ray (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 1d EMA13)
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Zero-cross signals: Bull Power crosses above zero (long), Bear Power crosses below zero (short)
    bull_cross_up = (bull_power[1:] > 0) & (bull_power[:-1] <= 0)
    bear_cross_down = (bear_power[1:] < 0) & (bear_power[:-1] >= 0)
    # Prepend False for index 0
    bull_cross_up = np.concatenate([[False], bull_cross_up])
    bear_cross_down = np.concatenate([[False], bear_cross_down])
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) + EMA13 (13) + volume MA (20)
    start_idx = 34 + 20  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation and trend alignment
            # Long: Bull Power crosses above zero with uptrend and volume confirmation
            long_signal = bull_cross_up[i] and uptrend and volume_confirm[i]
            # Short: Bear Power crosses below zero with downtrend and volume confirmation
            short_signal = bear_cross_down[i] and downtrend and volume_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if Bear Power crosses below zero (momentum shift) or trend changes
            if bear_cross_down[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if Bull Power crosses above zero (momentum shift) or trend changes
            if bull_cross_up[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0