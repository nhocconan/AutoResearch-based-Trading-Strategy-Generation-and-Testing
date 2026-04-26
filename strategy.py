#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: Use 1d Camarilla R1/S1 breakouts with weekly EMA trend filter and volume spike confirmation. The 1d timeframe reduces trade frequency to optimal range (7-25/year) while weekly trend filter ensures we only trade with the dominant trend, reducing whipsaws in both bull and bear markets. Volume spike adds confirmation of institutional participation. Position sizing at 0.25 for controlled drawdown.
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
    
    # Get 1d data for Camarilla calculation and weekly data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(21) on weekly for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate ATR(14) for stoploss on 1d
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
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
    
    # Align Camarilla levels to 1d
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d data needs, weekly EMA, volume MA, ATR
    start_idx = max(1, 21, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        weekly_uptrend = close_val > ema_21_1w_aligned[i]   # Weekly uptrend
        weekly_downtrend = close_val < ema_21_1w_aligned[i]  # Weekly downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 AND weekly trend up AND volume spike
            long_signal = (close_val > r1_aligned[i]) and weekly_uptrend and vol_spike
            
            # Short: price breaks below S1 AND weekly trend down AND volume spike
            short_signal = (close_val < s1_aligned[i]) and weekly_downtrend and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: weekly trend flips down OR price hits ATR stoploss
            if (not weekly_uptrend) or (close_val < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: weekly trend flips up OR price hits ATR stoploss
            if (not weekly_downtrend) or (close_val > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0