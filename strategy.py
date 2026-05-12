#!/usr/bin/env python3
# 4H_CHOPPINESS_REGIME_DONCHIAN_BREAKOUT
# Hypothesis: In choppy markets (CHOPPINESS > 61.8), mean-reversion works; in trending markets (CHOPPINESS < 38.2), breakout works.
# Use daily trend filter (EMA34) to avoid counter-trend trades. Entry: Donchian(20) breakout with volume confirmation.
# Works in bull and bear markets by adapting to regime and filtering with higher timeframe trend.
# Target: 20-40 trades/year on 4h timeframe.

name = "4H_CHOPPINESS_REGIME_DONCHIAN_BREAKOUT"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Choppiness Index on daily (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (n * (max_high - min_low))) / log10(n)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * (np.log10(atr14.rolling(window=14, min_periods=14).sum()) - 
                  np.log10(14 * (highest_high - lowest_low))) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Donchian channels (20-period) on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine regime
            is_choppy = chop_aligned[i] > 61.8
            is_trending = chop_aligned[i] < 38.2
            
            # Volume confirmation
            vol_ok = volume[i] > vol_ma20[i]
            
            if is_trending and vol_ok:
                # TRENDING: Breakout in direction of trend
                if close[i] > highest_high_20[i] and close[i] > ema34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low_20[i] and close[i] < ema34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif is_choppy and vol_ok:
                # CHOPPY: Mean reversion at Donchian edges
                if close[i] < lowest_low_20[i] and close[i] < ema34_aligned[i]:
                    signals[i] = 0.25  # Long at lower band in chop
                    position = 1
                elif close[i] > highest_high_20[i] and close[i] > ema34_aligned[i]:
                    signals[i] = -0.25  # Short at upper band in chop
                    position = -1
        elif position == 1:
            # EXIT LONG: Trend reversal or opposite signal
            if (close[i] <= ema34_aligned[i] or 
                (chop_aligned[i] < 38.2 and close[i] < lowest_low_20[i]) or  # Trending breakdown
                (chop_aligned[i] > 61.8 and close[i] > highest_high_20[i])):  # Chop reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or opposite signal
            if (close[i] >= ema34_aligned[i] or 
                (chop_aligned[i] < 38.2 and close[i] > highest_high_20[i]) or  # Trending breakout
                (chop_aligned[i] > 61.8 and close[i] < lowest_low_20[i])):  # Chop reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals