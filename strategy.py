#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Weekly Camarilla pivot (R4/S4) breakout on 6h with 1d EMA50 trend filter and volume spike confirmation. Uses discrete position sizing (0.25) to limit drawdown. Designed for low trade frequency (target 12-37/year) to overcome fee drag in ranging/bear markets like 2025+. Weekly pivots capture major support/resistance from smart money, while 1d EMA50 ensures we trade with the intermediate trend. Volume spike filters for institutional participation, reducing false breakouts. Works in bull markets via breakout continuation and in bear markets via fade at extremes (though primarily breakout-focused).
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for weekly Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate weekly Camarilla levels (R4/S4) from previous weekly bar
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1w['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1w['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1w['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r4 = pivot + (range_hl * 1.1 / 2.0)  # R4 = pivot + 1.1*range/2
    s4 = pivot - (range_hl * 1.1 / 2.0)  # S4 = pivot - 1.1*range/2
    
    # Align weekly Camarilla levels to 6h
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(50), volume MA, ATR
    start_idx = max(50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
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
        trend_1d_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R4 AND 1d trend up AND volume spike
            long_signal = (close_val > r4_aligned[i]) and trend_1d_up and vol_spike
            
            # Short: price breaks below S4 AND 1d trend down AND volume spike
            short_signal = (close_val < s4_aligned[i]) and trend_1d_down and vol_spike
            
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
            # Exit: trend flips down OR price hits ATR stoploss
            if (not trend_1d_up) or (close_val < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_1d_down) or (close_val > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0