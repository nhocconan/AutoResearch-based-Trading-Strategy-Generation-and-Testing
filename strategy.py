#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA50_1dVolumeSpike_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 1h with 4h EMA50 trend filter and 1d volume spike confirmation.
Uses ATR trailing stop (2.0x) and session filter (08-20 UTC) to reduce noise. Position size 0.20.
Designed for low trade frequency (target 15-37/year) to minimize fee drag while capturing breakouts
in both bull and bear markets via confluence: pivot break + HTF trend + volume spike + session filter.
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
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume median for spike confirmation (20-period lookback)
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels from previous 1h bar (using 1h OHLC)
    # We need to compute Camarilla from previous completed 1h bar
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align 4h EMA and 1d volume median to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    # ATR for stop (14-period on 1h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    bars_since_entry = 0
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Warmup: max of 4h EMA (50), 1d volume median (20), 1h ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_median_1d_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        vol_median_1d_val = vol_median_1d_aligned[i]
        camarilla_r1_val = camarilla_r1[i]
        camarilla_s1_val = camarilla_s1[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        atr_14_val = atr_14[i]
        
        if position == 0 and in_session:
            # Long: break above R1, uptrend (close > 4h EMA50), volume spike (>2.0x 1d median volume)
            long_signal = (high_val > camarilla_r1_val) and \
                          (close_val > ema_50_4h_val) and \
                          (volume_val > 2.0 * vol_median_1d_val)
            # Short: break below S1, downtrend (close < 4h EMA50), volume spike (>2.0x 1d median volume)
            short_signal = (low_val < camarilla_s1_val) and \
                           (close_val < ema_50_4h_val) and \
                           (volume_val > 2.0 * vol_median_1d_val)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.20
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < 4h EMA50) after minimum holding period
            if bars_since_entry >= 3 and ((low_val < long_stop) or (close_val < ema_50_4h_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.20
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > 4h EMA50) after minimum holding period
            if bars_since_entry >= 3 and ((high_val > short_stop) or (close_val > ema_50_4h_val)):
                signals[i] = 0.0
                position = 0
        else:
            # Outside session or flat: maintain flat or hold position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0