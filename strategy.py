#!/usr/bin/env python3
# 6h_1w_1d_cci_extreme_reversion
# Hypothesis: Weekly CCI extremes (>150 or <-150) indicate overbought/oversold conditions.
# On 6h timeframe, we mean-revert: short when weekly CCI >150 and 6h price closes below 1d VWAP,
# long when weekly CCI <-150 and 6h price closes above 1d VWAP.
# Uses 1-day VWAP as dynamic support/resistance. Designed for 15-30 trades/year to avoid fee drag.
# Works in both bull/bear markets by fading extremes at key daily levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_cci_extreme_reversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-week data for CCI extreme detection
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1-day data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate CCI on weekly data (20-period)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    sma_tp_1w = pd.Series(typical_price_1w).rolling(window=20, min_periods=20).mean().values
    mad_1w = pd.Series(typical_price_1w).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # Avoid division by zero
    cci_1w = np.where(mad_1w != 0, (typical_price_1w - sma_tp_1w) / (0.015 * mad_1w), 0.0)
    cci_1w_aligned = align_htf_to_ltf(prices, df_1w, cci_1w)
    
    # Calculate 1-day VWAP (typical price * volume cumsum / volume cumsum)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    pv_1d = typical_price_1d * volume_1d
    cum_pv_1d = np.cumsum(pv_1d)
    cum_volume_1d = np.cumsum(volume_1d)
    vwap_1d = np.where(cum_volume_1d != 0, cum_pv_1d / cum_volume_1d, typical_price_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(cci_1w_aligned[i]) or np.isnan(vwap_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1d VWAP or CCI extreme unwinds
            if close[i] < vwap_1d_aligned[i] or cci_1w_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 1d VWAP or CCI extreme unwinds
            if close[i] > vwap_1d_aligned[i] or cci_1w_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry at extremes
            # Long: weekly deeply oversold and price above 1d VWAP
            if cci_1w_aligned[i] < -150 and close[i] > vwap_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: weekly deeply overbought and price below 1d VWAP
            elif cci_1w_aligned[i] > 150 and close[i] < vwap_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals