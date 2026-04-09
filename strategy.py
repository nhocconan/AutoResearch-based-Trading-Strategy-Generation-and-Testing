#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with volume confirmation and ATR-based position sizing
# - Uses 1w HTF for Donchian channel (20-period high/low) to identify major trend breakouts
# - Enters long when price breaks above 1w Donchian high with volume > 1.5x 20-period average
# - Enters short when price breaks below 1w Donchian low with volume > 1.5x 20-period average
# - Uses ATR(14) for volatility-adjusted position sizing (0.20-0.30 range)
# - Includes choppiness filter to avoid ranging markets (CHOP > 61.8 = choppy, avoid entries)
# - Fixed holding period: exit after 10 bars to prevent overtrading and reduce fee drag
# - Target: 20-50 trades/year on 1d timeframe (80-200 total over 4 years)

name = "1d_1w_donchian_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channel (20 periods)
    period20_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, period20_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, period20_low)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR(14) for position sizing
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute choppiness index (14 periods) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = 100 * (np.log10(atr_14 / (range_14 + 1e-10)) / np.log10(14))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(chop[i]) or
            vol_ma_20[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Choppiness filter: avoid ranging markets (CHOP > 61.8 = choppy)
        not_choppy = chop[i] <= 61.8
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Volatility-adjusted position size (0.20 to 0.30 range)
        # Scale position size inversely with volatility to maintain consistent risk
        vol_scaling = np.clip(0.015 / (atr[i] / close[i] + 1e-10), 0.6, 1.2)
        base_size = 0.25
        position_size = base_size * vol_scaling
        position_size = np.clip(position_size, 0.20, 0.30)
        
        # Update bars since entry
        if position != 0:
            bars_since_entry += 1
        
        if position == 1:  # Long position
            # Exit conditions: time-based exit or breakdown
            if bars_since_entry >= 10 or close[i] < donchian_low_aligned[i]:
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: time-based exit or breakout
            if bars_since_entry >= 10 or close[i] > donchian_high_aligned[i]:
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation and regime filter
            if volume_confirmed and not_choppy:
                if breakout_up:
                    position = 1
                    bars_since_entry = 0
                    signals[i] = position_size
                elif breakout_down:
                    position = -1
                    bars_since_entry = 0
                    signals[i] = -position_size
    
    return signals