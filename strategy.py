#!/usr/bin/env python3
# 6h_ema_pullback_volume_4h1d_v1
# Hypothesis: 6h EMA pullback strategy with 4h trend filter and 1d volume regime.
# Uses 4h EMA(21) for trend direction, 6h EMA(8)/EMA(21) for pullback entries,
# and 1d volume spike (>2x 20-day average) to confirm institutional participation.
# Works in bull/bear by trading with higher timeframe trend only during high volume.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema_pullback_volume_4h1d_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA(21) for trend
    ema_21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d HTF data for volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # 1d volume spike: current volume > 2x 20-day average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # 6h EMA(8) and EMA(21) for pullback entries
    ema_8_6h = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema_21_6h = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_8_6h[i]) or
            np.isnan(ema_21_6h[i]) or np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: only trade when 1d volume spike is present
        high_volume_regime = vol_spike_aligned[i] > 0.5
        
        if position == 1:  # Long position
            # Exit: price crosses below 6h EMA(21) OR 4h trend turns bearish
            if close[i] < ema_21_6h[i] or ema_21_4h_aligned[i] < ema_8_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 6h EMA(21) OR 4h trend turns bullish
            if close[i] > ema_21_6h[i] or ema_21_4h_aligned[i] > ema_8_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if high_volume_regime:
                # Long entry: 4h bullish trend AND price pulls back to 6h EMA(8) from above
                if ema_21_4h_aligned[i] > ema_8_6h[i] and close[i] <= ema_8_6h[i] and close[i] > ema_21_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: 4h bearish trend AND price pulls back to 6h EMA(8) from below
                elif ema_21_4h_aligned[i] < ema_8_6h[i] and close[i] >= ema_8_6h[i] and close[i] < ema_21_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals