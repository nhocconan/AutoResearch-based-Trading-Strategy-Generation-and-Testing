#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility regime filter and volume confirmation
# Donchian breakout captures momentum in direction of higher timeframe volatility regime.
# ATR ratio (short/long) identifies low volatility squeezes that precede explosive moves.
# Volume spike confirms institutional participation. Designed for 20-30 trades/year on 4h to minimize fee drag.
# Works in bull markets via trend continuation and in bear markets via volatility expansion breakdowns.

name = "4h_Donchian20_ATRRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR ratio (7-period / 30-period) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    atr_7 = pd.Series(tr).ewm(span=7, adjust=False, min_periods=7).mean().values
    atr_30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    atr_ratio = atr_7 / atr_30  # >1 indicates expanding volatility
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient warmup for ATR calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(atr_ratio_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels using data up to current bar
        highest_high = np.max(high[i-19:i+1])  # 20-period high
        lowest_low = np.min(low[i-19:i+1])     # 20-period low
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Volatility regime filter: only trade when volatility is contracting (<1.0) or mildly expanding (<1.3)
        # Avoids choppy markets and waits for expansion after contraction
        vol_regime = atr_ratio_aligned[i] < 1.3
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high
        breakout_down = close[i] < lowest_low
        
        if position == 0:
            # Long: bullish breakout with volume spike and favorable volatility regime
            if breakout_up and volume_spike and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakdown with volume spike and favorable volatility regime
            elif breakout_down and volume_spike and vol_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint or volatility expands too much
            midpoint = (highest_high + lowest_low) / 2
            if close[i] < midpoint or atr_ratio_aligned[i] > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint or volatility expands too much
            midpoint = (highest_high + lowest_low) / 2
            if close[i] > midpoint or atr_ratio_aligned[i] > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals