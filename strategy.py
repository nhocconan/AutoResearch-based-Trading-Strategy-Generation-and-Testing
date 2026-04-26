#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R1 level AND 1d trend is up (close > EMA34) AND volume > 1.5x 20-period average. Enter short when price breaks below S1 level AND 1d trend is down (close < EMA34) AND volume spike. Only trade when market is not too choppy (Choppiness Index < 61.8). Uses Camarilla levels for precise support/resistance, 1d EMA34 for higher timeframe trend alignment, volume confirmation for institutional interest, and chop filter to avoid whipsaws in ranging markets. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Choppiness Index for regime filter
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).sum().values
    
    # Max/min high/low over 14 periods
    max_high_1d = high_1d.rolling(window=14, min_periods=14).max().values
    min_low_1d = low_1d.rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    range_1d = max_high_1d - min_low_1d
    chop_1d = np.where(
        range_1d > 0,
        100 * np.log10(atr_1d / range_1d) / np.log10(14),
        50  # neutral when range is zero
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Camarilla levels for 4h timeframe (based on previous day's OHLC)
    # Calculate daily OHLC from 1d data
    ohlc_1d = pd.DataFrame({
        'open': df_1d['open'].values,
        'high': df_1d['high'].values,
        'low': df_1d['low'].values,
        'close': df_1d['close'].values
    })
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # R1 = close + 1.0833*(high-low)/2, S1 = close - 1.0833*(high-low)/2
    prev_close = ohlc_1d['close'].shift(1).values
    prev_high = ohlc_1d['high'].shift(1).values
    prev_low = ohlc_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    R1 = prev_close + 1.0833 * prev_range / 2
    S1 = prev_close - 1.0833 * prev_range / 2
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 warmup (34), chop warmup (14), volume MA warmup (20)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade when not too choppy (Choppiness Index < 61.8)
        not_choppy = chop_1d_aligned[i] < 61.8
        
        if position == 0:
            # Long: price above R1 + 1d uptrend + volume spike + not choppy
            long_signal = (close[i] > R1_aligned[i]) and (close[i] > ema_34_1d_aligned[i]) and volume_spike[i] and not_choppy
            
            # Short: price below S1 + 1d downtrend + volume spike + not choppy
            short_signal = (close[i] < S1_aligned[i]) and (close[i] < ema_34_1d_aligned[i]) and volume_spike[i] and not_choppy
            
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
            # Exit: price breaks below S1 OR trend change to downtrend OR too choppy
            if (close[i] < S1_aligned[i]) or (close[i] < ema_34_1d_aligned[i]) or (not not_choppy):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend change to uptrend OR too choppy
            if (close[i] > R1_aligned[i]) or (close[i] > ema_34_1d_aligned[i]) or (not not_choppy):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0