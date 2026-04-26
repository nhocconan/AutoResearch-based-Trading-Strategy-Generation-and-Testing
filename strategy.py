#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ATRStop_v4
Hypothesis: Tighten entry conditions from v3 by requiring price to close beyond Camarilla R1/S1 (not just intraday touch) and adding a 2-bar confirmation filter to reduce whipsaw. This should lower trade frequency while maintaining edge in both bull (breakouts with trend) and bear (fade at extremes with volume exhaustion) markets. Uses discrete 0.30 sizing and ATR-based stoploss for risk control. Target: 20-40 trades/year per symbol to avoid fee drag.
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
    
    # Get 1d data for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 3.0 * 20-period average (tighter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (3.0 * vol_ma)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1d EMA(34), volume MA, ATR
    start_idx = max(34, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        close_val = close[i]
        close_prev = close[i-1]  # Prior bar close for confirmation
        trend_1d_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price closes above R1 AND prior bar was at/below R1 (breakout confirmation)
            #         AND 1d trend up AND volume spike
            long_signal = (close_val > r1_aligned[i]) and (close_prev <= r1_aligned[i]) and trend_1d_up and vol_spike
            
            # Short: price closes below S1 AND prior bar was at/above S1 (breakdown confirmation)
            #         AND 1d trend down AND volume spike
            short_signal = (close_val < s1_aligned[i]) and (close_prev >= s1_aligned[i]) and trend_1d_down and vol_spike
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            bars_since_entry += 1
            # Exit: trend flips down OR price hits ATR stoploss OR max 10 bars held (prevent stagnation)
            if (not trend_1d_up) or (close_val < entry_price - 2.0 * atr[i]) or (bars_since_entry >= 10):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            bars_since_entry += 1
            # Exit: trend flips up OR price hits ATR stoploss OR max 10 bars held
            if (not trend_1d_down) or (close_val > entry_price + 2.0 * atr[i]) or (bars_since_entry >= 10):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ATRStop_v4"
timeframe = "4h"
leverage = 1.0