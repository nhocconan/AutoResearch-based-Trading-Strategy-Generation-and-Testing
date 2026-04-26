#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ChopFilter_v2
Hypothesis: 4h Donchian(20) breakout confirmed by 1d EMA34 trend, volume spike, and choppiness regime filter. Uses discrete position sizing (0.25) to minimize fee drag. Designed for low trade frequency (<50/year) to avoid overtrading failures. Works in bull/bear via 1d trend filter and chop regime to avoid false signals in ranging markets.
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
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness Index (14-period) for regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    # We use trending regime only (CHOP < 38.2) to avoid whipsaws in ranging markets
    trending_regime = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 20 for Donchian, 34 for 1d EMA, 20 for volume avg, 14 for ATR/CHOP
    start_idx = max(20, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size (discrete level to reduce fee churn)
        
        if position == 0:
            # Flat - look for breakout with trend, volume, and regime confirmation
            # Long: break above Donchian high + 1d EMA34 uptrend + volume spike + trending regime
            long_entry = (close_val > donchian_high[i]) and \
                       (ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]) and \
                       volume_spike[i] and \
                       trending_regime[i]
            # Short: break below Donchian low + 1d EMA34 downtrend + volume spike + trending regime
            short_entry = (close_val < donchian_low[i]) and \
                        (ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]) and \
                        volume_spike[i] and \
                        trending_regime[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Donchian low break (trailing stop)
            if close_val < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Donchian high break (trailing stop)
            if close_val > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0