#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 12h Donchian(20) breakout in direction of 1d EMA34 trend with volume confirmation.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
Designed to work in both bull and bear markets via 1d trend filter: only take breakouts aligned with HTF trend.
ATH filter avoids buying near tops in extended bull runs. Volume confirmation reduces false breakouts.
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
    
    # Load 1d data ONCE before loop for HTF trend, Donchian channels, and ATH filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    dh_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 500-period high for ATH filter (avoid buying near all-time highs)
    high_500 = pd.Series(high).rolling(window=500, min_periods=500).max().values
    ath_filter = close < (0.9 * high_500)  # Only allow longs when price < 90% of 500-period high
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 20 for Donchian, 500 for ATH filter, 20 for volume MA)
    start_idx = max(34, 20, 500, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(dh_20_aligned[i]) or 
            np.isnan(dl_20_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(high_500[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        # Breakout conditions with trend filter and ATH filter
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long breakout above Donchian high with volume spike and ATH filter (avoid buying near tops)
            if close[i] > dh_20_aligned[i] and volume_spike and ath_filter[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Mean reversion short: breakdown below Donchian low in uptrend (fade the move)
            elif close[i] < dl_20_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1d
            # Short breakdown below Donchian low with volume spike
            if close[i] < dl_20_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Mean reversion long: breakout above Donchian high in downtrend (fade the move) with ATH filter
            elif close[i] > dh_20_aligned[i] and volume_spike and ath_filter[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0