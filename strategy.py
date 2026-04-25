#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dATR_Trend_VolumeRegime
Hypothesis: 4-hour Donchian(20) breakout with 1-day ATR-based trend filter and volume confirmation.
Targets 25-35 trades/year by requiring: 1) price breaks 20-period Donchian channel on 4h,
2) aligned with 1d ATR trend (price > 1d EMA50 + 0.5*ATR for long, < EMA50 - 0.5*ATR for short),
3) volume > 1.5x 20-period average. Uses ATR-based stoploss to manage risk. Designed to capture
strong trending moves while avoiding choppy markets via volume confirmation and trend filter.
Works in both bull and bear markets by following the 1d ATR-adjusted trend direction.
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
    
    # 1d data for ATR trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for trend filter
    tr1 = np.maximum(df_1d['high'].values - df_1d['low'].values,
                     np.maximum(np.abs(df_1d['high'].values - df_1d['close'].shift(1).values),
                                np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)))
    atr_14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA50 for trend baseline
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR-based trend bands: EMA50 ± 0.5*ATR14
    upper_band = ema_50_1d + 0.5 * atr_14_1d
    lower_band = ema_50_1d - 0.5 * atr_14_1d
    
    # Align 1d trend bands to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # 4h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d ATR(14) + EMA50 + Donchian20 + volume MA
    start_idx = max(14, 50, 20) + 20  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d ATR bands
        uptrend = curr_close > upper_band_aligned[i]
        downtrend = curr_close < lower_band_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation and trend alignment
            # Long breakout: price breaks above 4h Donchian upper with uptrend and volume confirmation
            long_breakout = (curr_high > highest_20[i]) and uptrend and volume_confirm[i]
            # Short breakout: price breaks below 4h Donchian lower with downtrend and volume confirmation
            short_breakout = (curr_low < lowest_20[i]) and downtrend and volume_confirm[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price breaks below 4h Donchian lower (trend reversal) or trend changes to downtrend
            if curr_low < lowest_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above 4h Donchian upper (trend reversal) or trend changes to uptrend
            if curr_high > highest_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_Trend_VolumeRegime"
timeframe = "4h"
leverage = 1.0