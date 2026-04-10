#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1w choppiness regime filter
# - Primary signal: Price breaks above/below 4h Donchian channel (20-period high/low)
# - Volume filter: 1d volume > 1.8x 20-period average volume (strong institutional participation)
# - Regime filter: 1w Choppiness Index > 61.8 (ranging market) for mean reversion at Donchian bands,
#                  or < 38.2 (trending market) for breakout continuation
# - In trending markets (CHOP < 38.2): breakout continuation in direction of break
# - In ranging markets (CHOP > 61.8): mean reversion - fade Donchian extremes toward midpoint
# - Neutral zone (38.2 <= CHOP <= 61.8): no trades to avoid whipsaw
# - Position size: 0.30 discrete level to balance return and fee drag
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(20)

name = "4h_1d_1w_donchian_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.8 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1w Choppiness Index
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) and sum of true ranges
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    chop_raw = np.where((hh_14 - ll_14) > 0,
                        100 * np.log10(tr_sum_14 / (hh_14 - ll_14)) / np.log10(14),
                        50)  # neutral when no range
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw, additional_delay_bars=0)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 4h ATR(20) for stoploss
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_20 = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(vol_spike_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: mean reversion to midpoint (ranging) OR stoploss hit
            if chop_aligned[i] > 61.8:  # ranging market - mean reversion
                if close_4h[i] < donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            elif chop_aligned[i] < 38.2:  # trending market - breakout continuation
                if close_4h[i] < donchian_low[i]:  # break below Donchian low invalidates
                    position = 0
                    signals[i] = 0.0
                elif close_4h[i] < entry_price - 2.0 * atr_20[i]:  # stoploss
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # neutral zone - maintain position
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: mean reversion to midpoint (ranging) OR stoploss hit
            if chop_aligned[i] > 61.8:  # ranging market - mean reversion
                if close_4h[i] > donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
            elif chop_aligned[i] < 38.2:  # trending market - breakout continuation
                if close_4h[i] > donchian_high[i]:  # break above Donchian high invalidates
                    position = 0
                    signals[i] = 0.0
                elif close_4h[i] > entry_price + 2.0 * atr_20[i]:  # stoploss
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
            else:  # neutral zone - maintain position
                signals[i] = -0.30
        else:  # Flat
            # Look for Donchian breakouts with volume filter and chop regime
            if vol_spike_aligned[i]:
                if chop_aligned[i] < 38.2:  # trending market - breakout continuation
                    # Long: price breaks above Donchian high
                    if close_4h[i] > donchian_high[i]:
                        position = 1
                        entry_price = close_4h[i]
                        signals[i] = 0.30
                    # Short: price breaks below Donchian low
                    elif close_4h[i] < donchian_low[i]:
                        position = -1
                        entry_price = close_4h[i]
                        signals[i] = -0.30
                elif chop_aligned[i] > 61.8:  # ranging market - mean reversion at extremes
                    # Long: price at Donchian low (support)
                    if close_4h[i] <= donchian_low[i] * 1.001:  # tiny buffer for noise
                        position = 1
                        entry_price = close_4h[i]
                        signals[i] = 0.30
                    # Short: price at Donchian high (resistance)
                    elif close_4h[i] >= donchian_high[i] * 0.999:
                        position = -1
                        entry_price = close_4h[i]
                        signals[i] = -0.30
    
    return signals