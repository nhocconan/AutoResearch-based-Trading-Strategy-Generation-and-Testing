#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Donchian upper band (20-period high) on 4h
# - Short when price breaks below Donchian lower band (20-period low) on 4h
# - Volume confirmation: 4h volume > 1.5x 20-period average
# - Regime filter: 1d Choppiness Index > 61.8 (range) enables mean reversion at bands
# - Discrete position sizing: 0.25 long/short to minimize fee drag
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)
# - Target: 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in bull/bear via regime adaptation: trend follow in trending, mean revert in range

name = "4h_1d_donchian_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for Choppiness Index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    
    # 1d Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (maxHH - minLL)) / log10(14)
    sum_atr = np.zeros_like(atr_14_1d)
    for i in range(14, len(sum_atr)):
        sum_atr[i] = np.sum(atr_14_1d[i-13:i+1])
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = max_hh - min_ll
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log10(sum_atr / denominator) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # fill NaN with neutral
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h Donchian channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR(14) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14_4h = np.zeros_like(tr_4h)
    atr_14_4h[14-1] = np.mean(tr_4h[:14])
    for i in range(14, len(tr_4h)):
        atr_14_4h[i] = (atr_14_4h[i-1] * (14-1) + tr_4h[i]) / 14
    
    # 4h volume confirmation: > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_4h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_4h[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price breaks below Donchian lower band
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] < donchian_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price breaks above Donchian upper band
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] > donchian_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume and regime filters
            if volume_spike[i]:
                # Regime adaptive logic:
                # CHOP > 61.8 = range (mean revert at bands)
                # CHOP < 38.2 = trending (trend follow breakouts)
                if chop_aligned[i] > 61.8:  # Range regime - mean reversion
                    # Long at lower band, Short at upper band
                    if prices['close'].iloc[i] <= donchian_low[i]:
                        position = 1
                        entry_price = prices['close'].iloc[i]
                        entry_atr = atr_14_4h[i]
                        signals[i] = 0.25
                    elif prices['close'].iloc[i] >= donchian_high[i]:
                        position = -1
                        entry_price = prices['close'].iloc[i]
                        entry_atr = atr_14_4h[i]
                        signals[i] = -0.25
                elif chop_aligned[i] < 38.2:  # Trending regime - follow breakouts
                    # Long on upper band break, Short on lower band break
                    if prices['close'].iloc[i] > donchian_high[i]:
                        position = 1
                        entry_price = prices['close'].iloc[i]
                        entry_atr = atr_14_4h[i]
                        signals[i] = 0.25
                    elif prices['close'].iloc[i] < donchian_low[i]:
                        position = -1
                        entry_price = prices['close'].iloc[i]
                        entry_atr = atr_14_4h[i]
                        signals[i] = -0.25
    
    return signals