#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_EMA_Trend_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with ZeroLag EMA trend filter and volume confirmation.
Long when Bull Power > 0 and price > ZeroLag EMA(34) with volume > 1.5x average.
Short when Bear Power < 0 and price < ZeroLag EMA(34) with volume > 1.5x average.
Uses ZeroLag EMA to reduce lag while maintaining trend accuracy.
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of Elder Ray signals, trend alignment, and volume.
Works in bull/bear via trend filter: only takes long in uptrend, short in downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Load 1d data ONCE before loop for HTF trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ZeroLag EMA(34) for trend filter
    ema_close = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    lag = pd.Series(ema_close).ewm(span=34, min_periods=34, adjust=False).mean().values
    zl_ema = 2 * ema_close - lag
    zl_ema_aligned = align_htf_to_ltf(prices, df_1d, zl_ema)  # Actually compute on 1d then align
    htf_trend = np.where(close > zl_ema_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Elder Ray from 1d data
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for ZeroLag EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(zl_ema_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume condition (moderate threshold)
        volume_ok = volume[i] > 1.5 * vol_ma_20[i]
        
        # Elder Ray conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long when Bull Power > 0 and price > ZeroLag EMA with volume
            if bull_power_aligned[i] > 0 and close[i] > zl_ema_aligned[i] and volume_ok:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if Bull Power turns negative
            elif position == 1 and bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1d
            # Short when Bear Power < 0 and price < ZeroLag EMA with volume
            if bear_power_aligned[i] < 0 and close[i] < zl_ema_aligned[i] and volume_ok:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if Bear Power turns positive
            elif position == -1 and bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ZeroLag_EMA_Trend_v1"
timeframe = "6h"
leverage = 1.0