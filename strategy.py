#!/usr/bin/env python3
# 12h_donchian_breakout_volume_chop_regime_v1
# Hypothesis: 12h Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and choppiness regime filter (CHOP(14) between 38.2 and 61.8 for ranging markets). Uses 1d HTF data for Donchian channels to reduce noise and improve signal quality. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-37 trades/year. Works in bull (breakouts) and bear (mean reversion in chop).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_chop_regime_v1"
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
    close_d = df_1d['close'].values
    
    # Daily Donchian(20) channels
    period20_high = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    donchian_high = period20_high
    donchian_low = period20_low
    
    # Align daily Donchian data to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 12h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP) on 12h - measures ranging vs trending markets
    # CHOP = 100 * log10(sum(ATR(1) over n) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # First TR
    atr1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1 / (np.log10(14) * (max_high - min_low)))
    
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
        
        # Regime filter: only trade in ranging markets (CHOP between 38.2 and 61.8)
        in_chop_zone = 38.2 <= chop[i] <= 61.8
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR volume fails OR exits chop zone
            if (close[i] < donchian_low_aligned[i] or not volume_confirmed or not in_chop_zone):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR volume fails OR exits chop zone
            if (close[i] > donchian_high_aligned[i] or not volume_confirmed or not in_chop_zone):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and in_chop_zone:
                # Long entry: price breaks above Donchian high
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals