#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v1
# Hypothesis: 4h strategy using Donchian channel breakout with volume confirmation and
# choppiness regime filter. Works in bull markets by catching breakouts and in bear
# markets by avoiding false signals during high volatility. Uses 1d HTF for choppiness
# index to determine ranging vs trending regimes. Discrete sizing (0.0, ±0.25) to
# minimize fee churn. Target: 25-35 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for choppiness index (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR and choppiness calculation
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Daily ATR(14) for choppiness denominator
    tr_d = np.maximum(np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1])),
                      np.abs(low_d[1:] - close_d[:-1]))
    tr_d = np.concatenate([[np.nan], tr_d])  # Align length
    atr_d = pd.Series(tr_d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Daily True Range sum over 14 periods
    tr_sum_d = pd.Series(tr_d).rolling(window=14, min_periods=14).sum().values
    
    # Daily max(high) - min(low) over 14 periods
    max_high_d = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    min_low_d = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    range_d = max_high_d - min_low_d
    
    # Choppiness Index: 100 * log10(tr_sum / range) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    chop_raw = 100 * np.log10(tr_sum_d / range_d) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # 4h Donchian Channel (20-period)
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # 4h Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low OR loses volume confirmation
            if close[i] < donchian_low[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR loses volume confirmation
            if close[i] > donchian_high[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and is_trending:
                # Long entry: price breaks above Donchian high
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals