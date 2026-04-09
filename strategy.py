#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-period average volume) and chop regime filter (CHOP(14) > 61.8 for mean reversion in ranging markets). Enters long on upper band breakout with volume and chop filter; short on lower band breakout with volume and chop filter. Exits on opposite Donchian band touch. Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 20-50 trades/year) to work in both bull and bear markets by capturing institutional breakouts in ranging regimes where false breakouts fade.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Chopiness Index (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where(
        (atr > 0) & (highest_high_14 > lowest_low_14),
        100 * np.log10(np.sum(tr[-14:]) / np.log(10) / (highest_high_14 - lowest_low_14)) / np.log10(14),
        50
    )
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(80, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Chop regime filter: CHOP > 61.8 indicates ranging market (mean reversion favorable)
        chop_filter = chop[i] > 61.8
        
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
            # Enter only with volume confirmation and chop regime filter
            if volume_confirmed and chop_filter:
                # Long: price breaks above upper Donchian band
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian band
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals