#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
Hypothesis: Combines Camarilla pivot breakout on 1h with 4h trend filter (EMA50) and 1d volume spike confirmation.
Uses 4h/1d for signal direction, 1h only for entry timing. Target: 15-35 trades/year to minimize fee drag.
Works in bull markets via breakouts with trend, in bear markets via mean reversion at Camarilla levels during low volatility.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h EMA50 for trend filter ---
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h.values)
    
    # --- 1d volume spike detection ---
    vol_ma_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_1d.values)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # --- Previous day Camarilla levels (using prior 1d OHLC) ---
    # Shift to ensure we use only completed daily data
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # --- Session filter: 08-20 UTC ---
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session, flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on 4h EMA50
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Breakout signals
        long_breakout = (high[i] > R1_aligned[i]) and vol_spike_1d_aligned[i]
        short_breakout = (low[i] < S1_aligned[i]) and vol_spike_1d_aligned[i]
        
        # Mean reversion signals (only in low volume environments)
        low_vol_env = not vol_spike_1d_aligned[i]  # No volume spike = low volatility
        long_reversion = (low[i] <= S1_aligned[i]) and low_vol_env and uptrend
        short_reversion = (high[i] >= R1_aligned[i]) and low_vol_env and downtrend
        
        if position == 0:
            if uptrend:
                # In uptrend, prioritize long breakouts and mean reversion at S1
                if long_breakout:
                    signals[i] = 0.20
                    position = 1
                elif long_reversion:
                    signals[i] = 0.20
                    position = 1
            elif downtrend:
                # In downtrend, prioritize short breakouts and mean reversion at R1
                if short_breakout:
                    signals[i] = -0.20
                    position = -1
                elif short_reversion:
                    signals[i] = -0.20
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price reaches R1 (profit target) or breaks S1 (stop)
                exit_signal = (low[i] <= S1_aligned[i]) or (high[i] >= R1_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price reaches S1 (profit target) or breaks R1 (stop)
                exit_signal = (high[i] >= R1_aligned[i]) or (low[i] <= S1_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals