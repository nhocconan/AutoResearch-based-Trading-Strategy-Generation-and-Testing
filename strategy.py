#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dATRRegime_VolumeSpike_ATRStop_v1
Hypothesis: Donchian(20) breakout on 4h with 1d ATR-based regime filter (trend if ATR rising, range if falling) and volume spike confirmation. Uses discrete position sizing (0.25) to balance return and drawdown. Designed for low trade frequency (target 20-50/year) to overcome fee drag in ranging/bear markets. Works in both bull (breakouts with rising volatility regime) and bear (fade at extremes with volume exhaustion) via volume spike filter that captures institutional participation. Added ATR stoploss (2.0x) to reduce trade frequency and improve Sharpe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # ATR regime: rising ATR = trending market, falling ATR = ranging market
    atr_ma_5_1d = pd.Series(atr_14_1d).rolling(window=5, min_periods=5).mean().values
    atr_regime_trending = atr_14_1d > atr_ma_5_1d  # True if ATR rising (trending)
    atr_regime_ranging = atr_14_1d <= atr_ma_5_1d   # True if ATR falling/ranging
    
    # Align 1d ATR regime to 4h
    atr_regime_trending_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_trending)
    atr_regime_ranging_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_ranging)
    
    # Calculate ATR(14) on 4h for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter: volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), volume MA, ATR
    start_idx = max(20, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(atr_regime_trending_aligned[i]) or
            np.isnan(atr_regime_ranging_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        regime_trending = atr_regime_trending_aligned[i]
        regime_ranging = atr_regime_ranging_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND volume spike AND (trending OR ranging)
            long_signal = (close_val > highest_20[i]) and vol_spike and (regime_trending or regime_ranging)
            
            # Short: price breaks below Donchian lower AND volume spike AND (trending OR ranging)
            short_signal = (close_val < lowest_20[i]) and vol_spike and (regime_trending or regime_ranging)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price hits ATR stoploss OR Donchian middle (mean reversion in ranging)
            if (close_val < entry_price - 2.0 * atr_14[i]) or \
               (regime_ranging and close_val < (highest_20[i] + lowest_20[i]) / 2):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price hits ATR stoploss OR Donchian middle (mean reversion in ranging)
            if (close_val > entry_price + 2.0 * atr_14[i]) or \
               (regime_ranging and close_val > (highest_20[i] + lowest_20[i]) / 2):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dATRRegime_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0