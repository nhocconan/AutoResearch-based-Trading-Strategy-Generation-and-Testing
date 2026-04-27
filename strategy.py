#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Choppiness Index regime filter with Donchian breakout entries.
# Uses daily Choppiness Index (14) to identify regime: >61.8 = range (mean revert),
# <38.2 = trending (trend follow). In trending regimes, enter Donchian(20) breakouts
# in direction of trend. In ranging regimes, fade at Donchian bands.
# Designed for low trade frequency (~15-30/year) with regime adaptation to work in
# both bull and bear markets by switching between trend following and mean reversion.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for Choppiness Index
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if i == 13:
            atr_14[i] = np.nanmean(tr[:14])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR14)/ (max(high)-min(low)) over 14 periods) / log10(14)
    sum_atr_14 = np.full(len(tr), np.nan)
    max_high_14 = np.full(len(tr), np.nan)
    min_low_14 = np.full(len(tr), np.nan)
    
    for i in range(13, len(tr)):
        sum_atr_14[i] = np.nansum(tr[i-13:i+1])
        max_high_14[i] = np.nanmax(high_1d[i-13:i+1])
        min_low_14[i] = np.nanmin(low_1d[i-13:i+1])
    
    chop = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if max_high_14[i] > min_low_14[i] and sum_atr_14[i] > 0:
            chop[i] = 100 * np.log10(sum_atr_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 6h Donchian channels (20-period)
    donch_h = np.full(n, np.nan)
    donch_l = np.full(n, np.nan)
    
    for i in range(19, n):
        donch_h[i] = np.max(high[i-19:i+1])
        donch_l[i] = np.min(low[i-19:i+1])
    
    # 6h EMA(20) for trend filter
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need chop (14), Donchian (20), EMA (20)
    start_idx = max(14, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_h[i]) or np.isnan(donch_l[i]) or
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        
        # Regime determination
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        # Neutral zone (38.2-61.8) - no trades
        
        if position == 0:
            if is_trending:
                # Trending regime: follow Donchian breakout with EMA filter
                if price > donch_h[i] and price > ema_20[i]:
                    signals[i] = size
                    position = 1
                elif price < donch_l[i] and price < ema_20[i]:
                    signals[i] = -size
                    position = -1
            elif is_ranging:
                # Ranging regime: mean reversion at Donchian bands
                if price < donch_l[i]:
                    signals[i] = size
                    position = 1
                elif price > donch_h[i]:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: opposite Donchian band or regime change to ranging with reversion signal
            if price > donch_h[i] or (is_ranging and price > donch_l[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: opposite Donchian band or regime change to ranging with reversion signal
            if price < donch_l[i] or (is_ranging and price < donch_h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Chop_Donchian_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0