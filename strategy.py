#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeSpike
Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 1-day EMA34 trend filter and volume confirmation.
Targets 12-30 trades/year by requiring: 1) Bull/Bear Power extreme (top/bottom 10% of 50-period), 
2) aligned with 1d EMA34 trend, 3) volume > 1.8x 20-period average. Uses 6h timeframe to reduce 
overtrading while capturing significant moves. Volume spike filter reduces false signals.
Designed to work in both bull and bear markets by following the 1d trend direction.
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
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d data for EMA13 (used in Elder Ray calculation)
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    # Using 1d EMA13 aligned to 6h timeframe
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Percentile thresholds for extreme Bull/Bear Power (top/bottom 10% over 50-period lookback)
    def rolling_percentile(arr, window, percentile):
        """Calculate rolling percentile using pandas Series"""
        return pd.Series(arr).rolling(window=window, min_periods=window).apply(
            lambda x: np.percentile(x, percentile), raw=True
        ).values
    
    bull_power_high_threshold = rolling_percentile(bull_power, 50, 90)  # Top 10%
    bear_power_low_threshold = rolling_percentile(bear_power, 50, 10)   # Bottom 10%
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) + EMA13 (13) + percentile (50) + volume MA (20)
    start_idx = 50 + 20  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_high_threshold[i]) or np.isnan(bear_power_low_threshold[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation and trend alignment
            # Long: Bull Power extreme (above 90th percentile) with uptrend and volume confirmation
            long_signal = (bull_power[i] > bull_power_high_threshold[i]) and uptrend and volume_confirm[i]
            # Short: Bear Power extreme (below 10th percentile) with downtrend and volume confirmation
            short_signal = (bear_power[i] < bear_power_low_threshold[i]) and downtrend and volume_confirm[i]
            
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
            # Long position: exit conditions
            # Exit if Bear Power becomes extreme (mean reversion) or trend changes to downtrend
            if (bear_power[i] < bear_power_low_threshold[i]) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if Bull Power becomes extreme (mean reversion) or trend changes to uptrend
            if (bull_power[i] > bull_power_high_threshold[i]) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0