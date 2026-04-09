#!/usr/bin/env python3
# 4h_donchian_volume_chop_v4
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-period MA) and
# choppiness regime filter (CHOP(14) between 38.2 and 61.8 for ranging markets).
# Long when price breaks above Donchian upper band in choppy/range regime with volume spike.
# Short when price breaks below Donchian lower band in choppy/range regime with volume spike.
# Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 20-40 trades/year.
# Daily HTF used only for choppiness calculation to avoid look-ahead.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily HTF data for choppiness calculation (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Calculate True Range for daily data
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) for daily
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (max(HH) - min(LL))) / log10(14)
    # where sum(ATR14) is over 14 periods, max(HH) is highest high over 14 periods,
    # min(LL) is lowest low over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_14 - min_low_14
    chop_denom = np.where(chop_denom == 0, np.nan, chop_denom)
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    chop_1d = chop_raw  # Already in correct orientation
    
    # Align daily choppiness to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 4h Donchian channels (20-period)
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # 4h volume confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in choppy/range markets (CHOP between 38.2 and 61.8)
        chop = chop_1d_aligned[i]
        if chop < 38.2 or chop > 61.8:
            # In trending market, stay flat to avoid whipsaws
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian lower band OR volume drops
            if close[i] < donchian_low[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper band OR volume drops
            if close[i] > donchian_high[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above Donchian upper band
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian lower band
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals