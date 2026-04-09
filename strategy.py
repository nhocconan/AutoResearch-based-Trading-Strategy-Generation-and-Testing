#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# In trending markets (chop < 38.2), breakouts capture momentum; in ranging markets (chop > 61.8),
# mean reversion at channel edges works. Volume confirmation filters false breakouts.
# Uses discrete sizing (0.0, ±0.30) to minimize fee churn. Target: 20-50 trades/year per symbol.

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
    
    # 1d HTF data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate choppiness index on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for ATR calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness index: 100 * log10(sum(TR14) / (log10(14) * (HH14 - LL14))) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = hh_14 - ll_14
    hh_ll_diff = np.where(hh_ll_diff == 0, 1e-10, hh_ll_diff)
    chop = 100 * np.log10(sum_tr_14) / (np.log10(14) * np.log10(hh_ll_diff)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channel (20-period) on 4h timeframe
    period = 20
    dc_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    dc_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Volume confirmation (20-period average)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian low or volume dries up
            if close[i] < dc_low[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian high or volume dries up
            if close[i] > dc_high[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            if volume_confirmed:
                chop_val = chop_aligned[i]
                # In trending markets (chop < 38.2): breakout strategy
                if chop_val < 38.2:
                    # Long breakout: price closes above Donchian high
                    if close[i] > dc_high[i]:
                        position = 1
                        signals[i] = 0.30
                    # Short breakout: price closes below Donchian low
                    elif close[i] < dc_low[i]:
                        position = -1
                        signals[i] = -0.30
                # In ranging markets (chop > 61.8): mean reversion at channel edges
                elif chop_val > 61.8:
                    # Long mean reversion: price touches Donchian low with rejection
                    if low[i] <= dc_low[i] and close[i] > dc_low[i]:
                        position = 1
                        signals[i] = 0.30
                    # Short mean reversion: price touches Donchian high with rejection
                    elif high[i] >= dc_high[i] and close[i] < dc_high[i]:
                        position = -1
                        signals[i] = -0.30
    
    return signals