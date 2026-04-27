#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_4hEMA20_Trend_VolumeSpike_Regime_v1
Hypothesis: Donchian(20) breakout on 4h with 4h EMA20 trend filter, volume confirmation, and choppiness regime filter.
Enters only when price breaks Donchian channel in direction of 4h EMA20 trend with volume spike and chop < 61.8 (trending regime).
Uses ATR-based stoploss for risk management. Designed for low trade frequency (target: 20-50/year) to minimize fee drag.
Works in both bull and bear markets by aligning with intermediate-term trend and regime filter.
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
    
    # Calculate ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h EMA20 for trend filter (using same timeframe as primary)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate choppiness index regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # Simplified: CHOP < 38.2 = trending, > 61.8 = ranging
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high - lowest_low
    # Avoid division by zero
    chop_raw = 100 * np.log10(sum_atr_14 / np.maximum(range_14, 1e-10)) / np.log10(14)
    chop_raw = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50.0, chop_raw)  # neutral when invalid
    chop_filter = chop_raw < 61.8  # trending regime when chop < 61.8
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for ATR, Donchian, EMA20, volume average, chop
    start_idx = max(100, 20, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_20[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_20[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        chop_regime = chop_filter[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry: Donchian breakout in direction of EMA20 trend with volume spike and trending regime
            # Long: price breaks above Donchian high AND EMA20 trend up (price > EMA20) AND volume spike AND chop < 61.8
            # Short: price breaks below Donchian low AND EMA20 trend down (price < EMA20) AND volume spike AND chop < 61.8
            long_breakout = close_val > donchian_high[i]
            short_breakout = close_val < donchian_low[i]
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            
            if long_breakout and trend_up and vol_spike and chop_regime:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout and trend_down and vol_spike and chop_regime:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Donchian low (failed breakout) or ATR stoploss hit
            if close_val < donchian_low[i] or close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian high (failed breakout) or ATR stoploss hit
            if close_val > donchian_high[i] or close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_4hEMA20_Trend_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0