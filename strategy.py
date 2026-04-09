#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long: Price breaks above Donchian(20) upper band with volume > 1.5x 20-period average and CHOP(14) > 61.8 (ranging market -> mean reversion long from lower band)
# Short: Price breaks below Donchian(20) lower band with volume > 1.5x 20-period average and CHOP(14) > 61.8 (ranging market -> mean reversion short from upper band)
# Exit: Price returns to Donchian midpoint.
# Uses 1d ATR for volatility filter: only trade when ATR(14) > 0.5 * ATR(50) to avoid low volatility periods.
# Target: 20-50 trades/year to minimize fee drag while maintaining edge.
# Donchian breakouts work in both bull (trend continuation) and bear (mean reversion in ranging markets) markets when combined with chop filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter (14 and 50 period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_50 = tr.rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_14 > (0.5 * atr_50)
    
    # Donchian channels (20-period)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Get 1d data for choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    close_1d_s = pd.Series(close_1d)
    tr1_1d = high_1d_s - low_1d_s
    tr2_1d = abs(high_1d_s - close_1d_s.shift(1))
    tr3_1d = abs(low_1d_s - close_1d_s.shift(1))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    
    # Choppiness Index (14-period)
    atr_sum_1d = tr_1d.rolling(window=14, min_periods=14).sum()
    hh_1d = high_1d_s.rolling(window=14, min_periods=14).max()
    ll_1d = low_1d_s.rolling(window=14, min_periods=14).min()
    chop_1d = 100 * np.log10(atr_sum_1d / (hh_1d - ll_1d)) / np.log10(14)
    chop_1d = chop_1d.values
    
    # Align 1d data to 4h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_ma[i]) or np.isnan(volume[i]) or np.isnan(close[i]) or
            np.isnan(volatility_filter[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Choppiness regime: CHOP > 61.8 indicates ranging market (mean reversion opportunity)
        chop_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian upper with volume and chop regime
            if (close[i] > donchian_upper[i] and    # Break above upper band
                volume_confirmed and                # Volume spike
                chop_regime and                     # Ranging market -> mean reversion long
                volatility_filter[i]):              # Sufficient volatility
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower with volume and chop regime
            elif (close[i] < donchian_lower[i] and  # Break below lower band
                  volume_confirmed and              # Volume spike
                  chop_regime and                   # Ranging market -> mean reversion short
                  volatility_filter[i]):            # Sufficient volatility
                position = -1
                signals[i] = -0.25
    
    return signals