#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike
Hypothesis: Camarilla R1/S1 breakouts on 1h with 4h EMA50 trend filter and 1d volume spike confirmation.
Long when price breaks above R1, 4h EMA50 up, 1d volume > 1.5x average.
Short when price breaks below S1, 4h EMA50 down, 1d volume > 1.5x average.
Uses ATR-based trailing stop (2.0x ATR) for risk management.
Designed for 1h timeframe with tight entries (15-37/year) to avoid fee drag while capturing breakouts in both bull and bear markets.
Uses discrete position sizing (0.20) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume average for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Camarilla levels for 1h (using previous day's OHLC)
    # We'll approximate using rolling window of 24*4 = 96 periods (previous day)
    lookback = 96  # 24 hours * 4 (1h periods per 4h) = 96 for previous day
    
    # Calculate pivot and levels using previous day's OHLC
    # For each bar, use OHLC from 96 bars ago to simulate previous day
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 4h EMA (50), 1d volume MA (20), lookback (96)
    start_idx = max(50, 20, 96)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        
        # Get previous day's OHLC (96 bars ago for 1h data)
        prev_idx = i - lookback
        if prev_idx < 0:
            # Not enough data for previous day, hold
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
            
        prev_high = high[prev_idx:prev_idx+lookback].max()
        prev_low = low[prev_idx:prev_idx+lookback].min()
        prev_close = close[prev_idx:prev_idx+lookback].mean()  # approximate close
        
        # Calculate Camarilla levels
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        r1 = pivot + (range_val * 1.1 / 12)
        s1 = pivot - (range_val * 1.1 / 12)
        
        # Volume spike condition: current 1h volume > 1.5 * 1d average volume
        volume_spike = volume_val > 1.5 * vol_ma_1d_val
        
        if position == 0:
            # Long: price breaks above R1, 4h EMA50 up, volume spike
            long_signal = (close_val > r1) and (ema_50_4h_val > ema_50_4h_aligned[i-1]) and volume_spike
            # Short: price breaks below S1, 4h EMA50 down, volume spike
            short_signal = (close_val < s1) and (ema_50_4h_val < ema_50_4h_aligned[i-1]) and volume_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * ((high_val - low_val) * 0.5)  # approximate ATR
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * ((high_val - low_val) * 0.5)  # approximate ATR
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Update trailing stop: move stop up as price makes new highs
            atr_approx = (high_val - low_val) * 0.5
            long_stop = max(long_stop, high_val - 2.0 * atr_approx)
            # Exit: trailing stop hit or trend reversal
            if (low_val < long_stop) or (ema_50_4h_val < ema_50_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Update trailing stop: move stop down as price makes new lows
            atr_approx = (high_val - low_val) * 0.5
            short_stop = min(short_stop, low_val + 2.0 * atr_approx)
            # Exit: trailing stop hit or trend reversal
            if (high_val > short_stop) or (ema_50_4h_val > ema_50_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0