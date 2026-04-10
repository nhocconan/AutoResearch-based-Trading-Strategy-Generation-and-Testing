#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Donchian(20) high + volume spike + chop < 61.8 (trending)
# - Short when price breaks below Donchian(20) low + volume spike + chop < 61.8 (trending)
# - Uses ATR-based trailing stop: exit when price moves against position by 2.5x ATR(14)
# - Discrete position sizing: 0.25 to minimize fee churn
# - Targets 20-40 trades/year (80-160 total over 4 years) to avoid fee drag
# - Works in both bull (trend continuation) and bear (trend continuation on shorts)

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d < 100):
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d volume confirmation: > 1.8x 20-period average (strict to reduce trades)
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (max(high,n) - min(low,n))) / log10(n)
    # We approximate: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # Using 14-period chop on daily
    atr_14 = atr_14_1d  # already computed
    sum_atr_14 = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        sum_atr_14[i] = np.sum(atr_14[i-13:i+1])
    sum_atr_14[:14] = np.nan
    
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    chop_14 = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        if range_14[i] > 0:
            chop_14[i] = 100 * np.log10(sum_atr_14[i] / range_14[i]) / np.log10(14)
        else:
            chop_14[i] = 50  # neutral when no range
    chop_14[:14] = np.nan
    chop_14_aligned = align_htf_to_ltf(prices, df_1d, chop_14)
    
    # 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        close_4h = prices['close'].iloc[i]
        
        # Skip if any required data is invalid
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(chop_14_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based trailing stop
            if close_4h < entry_price - 2.5 * entry_atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based trailing stop
            if close_4h > entry_price + 2.5 * entry_atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume spike and trending regime (CHOP < 61.8)
            if vol_spike_1d_aligned[i] and chop_14_aligned[i] < 61.8:
                # Long signal: break above Donchian high
                if close_4h > donchian_high[i]:
                    position = 1
                    entry_price = close_4h
                    entry_atr = atr_14_4h[i]
                    signals[i] = 0.25
                # Short signal: break below Donchian low
                elif close_4h < donchian_low[i]:
                    position = -1
                    entry_price = close_4h
                    entry_atr = atr_14_4h[i]
                    signals[i] = -0.25
    
    return signals