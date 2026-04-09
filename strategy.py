#!/usr/bin/env python3
# 1d_donchian_breakout_volume_chop_regime_v2
# Hypothesis: Daily Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and weekly choppiness regime filter (CHOP > 61.8 = range, mean reversion; CHOP < 38.2 = trending, trend follow). In ranging markets (CHOP > 61.8), fade Donchian breakouts (short upper band, long lower band). In trending markets (CHOP < 38.2), follow Donchian breakouts (long upper band, short lower band). Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 20-50 trades/year) to work in both bull and bear markets by adapting to regime.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_chop_regime_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period) on daily timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Weekly HTF data for choppiness regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly choppiness index (CHOP)
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(N)
    # where ATR(1) = TR = max(high-low, |high-close_prev|, |low-close_prev|)
    # Simplified: use true range over 14 periods
    atr_period = 14
    tr1 = np.maximum(df_1w['high'].values - df_1w['low'].values,
                     np.maximum(np.abs(df_1w['high'].values - np.roll(df_1w['close'].values, 1)),
                                np.abs(df_1w['low'].values - np.roll(df_1w['close'].values, 1))))
    # Handle first bar
    tr1[0] = df_1w['high'].values[0] - df_1w['low'].values[0]
    atr_values = pd.Series(tr1).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(df_1w['high'].values).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(df_1w['low'].values).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_val = max_high - min_low
    chop_raw = np.where(range_val > 0, 
                        100 * np.log10(pd.Series(atr_values).sum() / range_val) / np.log10(atr_period),
                        50.0)  # neutral when range is zero
    chop_values = chop_raw.fillna(50.0).values if isinstance(chop_raw, pd.Series) else np.where(np.isnan(chop_raw), 50.0, chop_raw)
    
    # Align weekly CHOP to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches or breaks lower Donchian band
            if close[i] <= lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks upper Donchian band
            if close[i] >= highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Regime-based logic
                chop_value = chop_aligned[i]
                
                if chop_value > 61.8:  # Ranging market - mean reversion
                    # Fade breakouts: short upper band, long lower band
                    if close[i] >= highest_high[i]:
                        position = -1
                        signals[i] = -0.25
                    elif close[i] <= lowest_low[i]:
                        position = 1
                        signals[i] = 0.25
                elif chop_value < 38.2:  # Trending market - trend follow
                    # Follow breakouts: long upper band, short lower band
                    if close[i] >= highest_high[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] <= lowest_low[i]:
                        position = -1
                        signals[i] = -0.25
                # In neutral regime (38.2 <= CHOP <= 61.8), no new entries
    
    return signals