#!/usr/bin/env python3
# 4h_donchian_breakout_1d_volume_chop_v2
# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d chop regime filter.
# Works in bull/bear: Donchian captures breakouts, volume confirms validity, chop filter avoids whipsaws in ranging markets.
# Target: 20-50 trades/year (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_volume_chop_v2"
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
    
    # 1d HTF data for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: current volume > 1.5x 20-period average
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d chop regime: CHOP(14) > 61.8 = ranging (avoid breakouts), CHOP < 38.2 = trending (favor breakouts)
    # True Range = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # approximate
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # approximate
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    atr_1d_safe = np.where(atr_1d == 0, np.nan, atr_1d)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_tr_14 / (atr_1d_safe * 14)) / np.log10(10)
    
    # Align 1d indicators to 4h timeframe (completed 1d bar only)
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR chop too high (ranging)
            if close[i] < donchian_low[i] or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR chop too high (ranging)
            if close[i] > donchian_high[i] or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and favorable chop regime (trending)
            volume_confirmed = volume[i] > 1.5 * volume_ma_1d_aligned[i]
            chop_favorable = chop_1d_aligned[i] < 38.2  # trending regime
            
            if volume_confirmed and chop_favorable:
                # Long: price breaks above Donchian high
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals