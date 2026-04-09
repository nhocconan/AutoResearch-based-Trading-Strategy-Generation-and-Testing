#!/usr/bin/env python3
# 12h_donchian_breakout_volume_chop_v1
# Hypothesis: 12h strategy using Donchian(20) breakout from daily timeframe for structure,
# with volume confirmation (>1.5x 20-period average) and choppiness regime filter (CHOP > 61.8 = range).
# Long when price breaks above Donchian high in ranging market (mean reversion bias).
# Short when price breaks below Donchian low in ranging market.
# Daily Donchian levels act as support/resistance; 12h timing captures momentum.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    
    # Daily Donchian(20) - structure from higher timeframe
    period20_high = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, period20_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, period20_low)
    
    # 12h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 12h choppiness index for regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest high - lowest low over 14))) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    high_low = np.maximum(high, low)
    high_low_shift = np.roll(high_low, 1)
    high_low_shift[0] = high_low[0]
    tr = np.maximum(high_low - low, np.maximum(high_low_shift - close, np.maximum(high_low_shift - high, 0)))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop_raw = 100 * np.log10(atr * atr_period / (highest_high - lowest_low + 1e-10)) / np.log10(atr_period)
    chop = pd.Series(chop_raw).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low OR chop drops below 50 (trending)
            if close[i] < donchian_low_aligned[i] or chop[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR chop drops below 50 (trending)
            if close[i] > donchian_high_aligned[i] or chop[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and ranging_market:
                # Long entry: price breaks above Donchian high in ranging market
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low in ranging market
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals