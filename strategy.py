#!/usr/bin/env python3
"""
12h_adaptive_risk_reversal_v1
Hypothesis: On 12h timeframe, use weekly volatility regime to toggle between mean-reversion at weekly Bollinger Bands (2 std) in low volatility and breakout of weekly Donchian channels (20-period) in high volatility. Volume confirmation filters false signals. Works in both bull and bear markets by adapting to volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_adaptive_risk_reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for regime and levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly close for calculations
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Bollinger Bands (20, 2) for mean reversion zone
    close_s_1w = pd.Series(close_1w)
    bb_mid = close_s_1w.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s_1w.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Weekly Donchian Channel (20) for breakout
    dc_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly volatility regime: BB width percentile
    bb_width = bb_upper - bb_lower
    # Use 50-period lookback for percentile ranking
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Align weekly levels to 12h timeframe
    bb_upper_12h = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_12h = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_mid_12h = align_htf_to_ltf(prices, df_1w, bb_mid)
    dc_high_12h = align_htf_to_ltf(prices, df_1w, dc_high)
    dc_low_12h = align_htf_to_ltf(prices, df_1w, dc_low)
    bb_width_pct_12h = align_htf_to_ltf(prices, df_1w, bb_width_percentile)
    
    # 20-period volume average on 12h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(bb_upper_12h[i]) or np.isnan(bb_lower_12h[i]) or 
            np.isnan(dc_high_12h[i]) or np.isnan(dc_low_12h[i]) or
            np.isnan(bb_width_pct_12h[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        # Volatility regime: low volatility if BB width percentile < 0.3
        low_vol_regime = bb_width_pct_12h[i] < 0.3
        high_vol_regime = bb_width_pct_12h[i] >= 0.7
        
        if position == 1:  # Long position
            # Exit conditions
            if low_vol_regime:
                # In low vol: exit if price returns to BB middle (mean reversion complete)
                if close[i] >= bb_mid_12h[i]:
                    position = 0
                    signals[i] = 0.0
            else:
                # In high vol: exit if price breaks below Donchian low
                if close[i] < dc_low_12h[i]:
                    position = 0
                    signals[i] = 0.0
            if position == 1:  # Still long
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit conditions
            if low_vol_regime:
                # In low vol: exit if price returns to BB middle
                if close[i] <= bb_mid_12h[i]:
                    position = 0
                    signals[i] = 0.0
            else:
                # In high vol: exit if price breaks above Donchian high
                if close[i] > dc_high_12h[i]:
                    position = 0
                    signals[i] = 0.0
            if position == -1:  # Still short
                signals[i] = -0.25
        else:  # Flat, look for entry
            if low_vol_regime and vol_confirm:
                # Low volatility: mean reversion at Bollinger Bands
                if close[i] <= bb_lower_12h[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= bb_upper_12h[i]:
                    position = -1
                    signals[i] = -0.25
            elif high_vol_regime and vol_confirm:
                # High volatility: breakout of Donchian channels
                if close[i] >= dc_high_12h[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] <= dc_low_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals