#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeConfirm_ChopRegime
Hypothesis: Donchian channel breakout on 4h with volume confirmation (>1.5x average volume) and choppiness regime filter (CHOP < 45 for trending markets). Uses discrete position sizing (0.25) to minimize fee churn. Designed to capture strong momentum moves in both bull and bear markets while avoiding whipsaws in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR1) / (max(high) - min(low))) / log10(N)
    atr_1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # True Range itself
    sum_atr_1 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_numerator = sum_atr_1
    chop_denominator = max_high - min_low
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_ratio = chop_numerator / chop_denominator
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We want strongly trending markets: CHOP < 45 (stricter for fewer trades)
    chop_filter = chop < 45
    
    # Calculate Donchian Channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    atr_multiplier = 2.5  # ATR stoploss multiplier
    
    # Start after warmup (need 20 for Donchian, 14 for ATR, 20 for volume, 14 for CHOP)
    start_idx = max(20, 14, 20, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        atr_val = atr[i]
        chop_val = chop_filter[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        
        # Skip if any data not ready
        if np.isnan(upper_channel) or np.isnan(lower_channel) or np.isnan(avg_vol) or np.isnan(atr_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Regime filter: only trade in strongly trending markets (CHOP < 45)
        regime_ok = chop_val
        
        # Long logic: price breaks above upper Donchian channel with volume confirmation and trending regime
        long_condition = (close_val > upper_channel) and volume_confirmed and regime_ok
        # Short logic: price breaks below lower Donchian channel with volume confirmation and trending regime
        short_condition = (close_val < lower_channel) and volume_confirmed and regime_ok
        
        # Exit logic: price retracement to middle of channel (50% level) OR regime change to ranging
        mid_channel = (upper_channel + lower_channel) / 2
        exit_long = close_val < mid_channel or not chop_val
        exit_short = close_val > mid_channel or not chop_val
        
        # ATR-based stoploss
        if position == 1:
            stop_price = entry_price - atr_multiplier * atr_val
            if close_val < stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            stop_price = entry_price + atr_multiplier * atr_val
            if close_val > stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeConfirm_ChopRegime"
timeframe = "4h"
leverage = 1.0