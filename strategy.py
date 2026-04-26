#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopFilter
Hypothesis: On 4h timeframe, Donchian(20) breakouts with volume confirmation (>1.5x 20-bar MA) and choppiness regime filter (CHOP > 61.8 for mean reversion, < 38.2 for trend) produce high-quality signals. In choppy markets, we mean-revert at Donchian extremes; in trending markets, we follow breakouts. Uses discrete sizing (0.0, ±0.30) to minimize fee churn. Targets 20-50 trades/year to avoid fee drag on 4h timeframe.
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high,n) - min(low,n))) / log10(n)
    atr_1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # ATR(1) = true range
    sum_atr_1 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = 100 * np.log10(sum_atr_1 / np.maximum(range_14, 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), ATR(14), volume MA(20), CHOP(14)
    start_idx = max(20, 14, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        close_val = close[i]
        vol_confirmed = vol_ratio[i] > 1.5  # volume at least 1.5x average
        chop_high = chop[i] > 61.8  # choppy regime (mean revert)
        chop_low = chop[i] < 38.2   # trending regime (follow trend)
        
        if position == 0:
            # In choppy regime: mean reversion at Donchian extremes
            # Long: price touches or breaks below Donchian low AND volume confirmation
            long_signal = (close_val <= donchian_low[i]) and vol_confirmed and chop_high
            # Short: price touches or breaks above Donchian high AND volume confirmation
            short_signal = (close_val >= donchian_high[i]) and vol_confirmed and chop_high
            
            # In trending regime: follow breakouts
            # Long: price breaks above Donchian high AND volume confirmation
            long_signal_trend = (close_val > donchian_high[i]) and vol_confirmed and chop_low
            # Short: price breaks below Donchian low AND volume confirmation
            short_signal_trend = (close_val < donchian_low[i]) and vol_confirmed and chop_low
            
            if long_signal or long_signal_trend:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif short_signal or short_signal_trend:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price reaches opposite Donchian band OR ATR stoploss
            if (close_val >= donchian_high[i]) or (close_val < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price reaches opposite Donchian band OR ATR stoploss
            if (close_val <= donchian_low[i]) or (close_val > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0