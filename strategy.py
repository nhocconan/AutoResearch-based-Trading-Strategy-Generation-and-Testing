#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R1 level AND 1d trend is up (close > EMA34) AND volume > 1.5x 20-period average volume AND market is not in extreme chop (choppiness index < 61.8). Enter short when price breaks below Camarilla S1 level AND 1d trend is down (close < EMA34) AND volume > 1.5x 20-period average volume AND market is not in extreme chop. Uses discrete sizing (0.0, ±0.25) to limit fee drift. Target: 12-30 trades/year. The chop filter avoids whipsaws in ranging markets, improving performance in both bull and bear regimes.
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
    
    # Get 1d data for Camarilla levels, trend filter, and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # need enough for EMA34 and chop calculation
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels from previous 1d bar (HLC of completed 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_raw = df_1d['close'].values  # raw 1d close for Camarilla calculation
    
    # Using previous completed 1d bar to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d_raw, 1)
    
    # First bar has no previous bar, set to NaN
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12
    s1 = prev_close_1d - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: fixed threshold of 1.5x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma
    
    # Choppiness index regime filter (14-period) on 1d timeframe
    # Chop > 61.8 indicates ranging/choppy market (avoid breakouts)
    # Chop < 38.2 indicates trending market (favor breakouts)
    # We use chop < 61.8 to avoid extreme chop, but allow trending and moderate chop
    true_range = np.maximum(df_1d['high'].values - df_1d['low'].values,
                            np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                                       np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
    true_range[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first bar
    
    atr14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    sum_tr14 = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14)
    chop[np.isnan(chop) | (atr14 == 0)] = 50  # default to neutral when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    not_extreme_chop = chop_aligned < 61.8  # avoid extreme ranging markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup, volume MA warmup, and chop warmup
    start_idx = max(34, 20, 34)  # EMA34 needs 34, volume MA needs 20, chop needs 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume spike + 1d uptrend + not extreme chop
            long_signal = breakout_up and volume_spike[i] and trend_uptrend and not_extreme_chop[i]
            
            # Short: breakout below S1 + volume spike + 1d downtrend + not extreme chop
            short_signal = breakout_down and volume_spike[i] and trend_downtrend and not_extreme_chop[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below R1 OR trend change to downtrend OR enter extreme chop
            if close[i] < r1_aligned[i] or not trend_uptrend or not not_extreme_chop[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above S1 OR trend change to uptrend OR enter extreme chop
            if close[i] > s1_aligned[i] or not trend_downtrend or not not_extreme_chop[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0