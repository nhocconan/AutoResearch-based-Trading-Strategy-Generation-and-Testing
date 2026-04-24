#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly ATR regime filter and volume confirmation.
- 6h timeframe targets 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.
- Uses weekly ATR to define volatility regime: high ATR = trend-following, low ATR = mean-reversion.
- In high volatility (weekly ATR > 1.5x 8-week mean): trade Donchian breakouts in direction of 12h EMA50.
- In low volatility: fade Donchian touches at bands with 12h EMA50 as dynamic support/resistance.
- Volume confirmation (>1.8x 20-period mean) ensures conviction on breakouts/fades.
- Works in bull/bear via ATR regime adaptation and EMA trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 60 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly ATR(14) for regime detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w[1:] - low_1w[:-1]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_mean_8_1w = pd.Series(atr_14_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    atr_ratio = atr_14_1w / atr_mean_8_1w
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # 6h Donchian(20) channels
    donch_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_m = (donch_h + donch_l) / 2
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_h[i]) or np.isnan(donch_l[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime: high volatility (trending) if weekly ATR > 1.5x mean
        high_vol_regime = atr_ratio_aligned[i] > 1.5
        
        if position == 0:
            if high_vol_regime:
                # High volatility: trend-following breakouts
                if close[i] > donch_h[i] and volume_spike[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donch_l[i] and volume_spike[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Low volatility: mean-reversion fades
                if close[i] <= donch_l[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                elif close[i] >= donch_h[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit conditions
            if high_vol_regime:
                # Exit trend long on Donchian middle cross or EMA cross
                if close[i] < donch_m[i] or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit mean-reversion long at Donchian upper band or EMA cross
                if close[i] >= donch_h[i] or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Short exit conditions
            if high_vol_regime:
                # Exit trend short on Donchian middle cross or EMA cross
                if close[i] > donch_m[i] or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit mean-reversion short at Donchian lower band or EMA cross
                if close[i] <= donch_l[i] or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "6h_Donchian20_weeklyATRRegime_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0