#!/usr/bin/env python3
# 6h_elder_ray_regime_v4
# Hypothesis: 6h strategy using Elder Ray (Bull/Bear Power) from 1d EMA13 with regime filter (ADX<20 ranging).
# In ranging markets (2025+), Elder Ray extremes signal mean reversion: short when Bull Power peaks, long when Bear Power troughs.
# Volume confirmation and discrete sizing (0.0, ±0.25) reduce fee churn. Target: 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_regime_v4"
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
    
    # 1d HTF data for Elder Ray and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA13 for Elder Ray
    close_s = pd.Series(close_1d)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 1d ADX(14) for regime filter
    plus_dm = np.diff(high_1d, prepend=high_1d[0])
    minus_dm = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_safe
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_safe
    dx = (np.abs(plus_di - minus_di) / (np.maximum(plus_di, minus_di) + 1e-10)) * 100
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Regime: only trade when market is ranging (ADX < 20)
        chop_regime = adx_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit: Bear Power improves (less negative) or volume dries up
            if bear_power_aligned[i] > -0.5 * np.std(bear_power_aligned[max(0, i-50):i+1]) or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power deteriorates (less positive) or volume dries up
            if bull_power_aligned[i] < 0.5 * np.std(bull_power_aligned[max(0, i-50):i+1]) or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: Bear Power at extreme low (strong bearish pressure likely to reverse)
                if bear_power_aligned[i] < np.percentile(bear_power_aligned[max(0, i-100):i+1], 5):
                    position = 1
                    signals[i] = 0.25
                # Short entry: Bull Power at extreme high (strong bullish pressure likely to reverse)
                elif bull_power_aligned[i] > np.percentile(bull_power_aligned[max(0, i-100):i+1], 95):
                    position = -1
                    signals[i] = -0.25
    
    return signals