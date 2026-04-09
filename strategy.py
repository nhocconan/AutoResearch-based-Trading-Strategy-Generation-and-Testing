#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
# In trending markets, breakouts capture momentum; in ranging markets (2025+), chop filter avoids false breakouts.
# Volume confirmation ensures breakouts have conviction. Discrete sizing (0.0, ±0.30) minimizes fee churn.
# Target: 75-200 total trades over 4 years by requiring Donchian breakout + volume spike + chop > 61.8 (trending regime).
# Primary timeframe: 4h, HTF: 12h for regime filter to avoid look-ahead.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v1"
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
    
    # 12h HTF data for choppiness regime filter (to avoid look-ahead)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness index regime filter (14-period) on 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True range calculation for ATR
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]  # first bar
    tr2[0] = 0
    tr3[0] = 0
    atr_14 = pd.Series(np.maximum(tr1, np.maximum(tr2, tr3))).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness index: CHOP = 100 * log10(sum(TR14) / log10(ATR(14))) / log10(14)
    high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict to reduce trades)
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        
        # Regime filter: only trade when market is trending (chop < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian low or volume dries up
            if close[i] < donchian_low[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian high or volume dries up
            if close[i] > donchian_high[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            if volume_confirmed and trending_regime:
                # Long entry: price breaks above Donchian high with volume confirmation
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.30
                # Short entry: price breaks below Donchian low with volume confirmation
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals