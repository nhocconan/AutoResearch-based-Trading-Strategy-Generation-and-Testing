#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and chop regime filter.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and chop regime.
- Donchian breakout: Close > upper band (long) or Close < lower band (short).
- Trend filter: Only trade breakouts in direction of 1d EMA50 (long if close > EMA50, short if close < EMA50).
- Volume confirmation: Volume > 1.5x 20-period volume MA.
- Chop regime filter: Only trade when 1d chop < 61.8 (trending regime).
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian bands, EMA trend, and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian bands (20-period) from prior 1d data
    # Upper band = max(high, 20), Lower band = min(low, 20)
    donchian_upper = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Chopiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values - pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    tr2 = abs(pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values - pd.Series(df_1d['close']).shift(1).rolling(window=14, min_periods=14).mean().values)
    tr3 = abs(pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values - pd.Series(df_1d['close']).shift(1).rolling(window=14, min_periods=14).mean().values)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (max_high14 - min_low14 + 1e-10)) / np.log10(14)
    chop_regime = chop < 61.8  # Trending regime when chop < 61.8
    
    # Align 1d indicators to 4h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(chop_regime_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Donchian breakout with volume spike, trend filter, and chop regime
            if volume_spike[i] and chop_regime_aligned[i]:
                # Long breakout: close > upper band and close > 1d EMA50 (uptrend)
                if close[i] > donchian_upper_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < lower band and close < 1d EMA50 (downtrend)
                elif close[i] < donchian_lower_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Donchian bands or opposite signal
            if close[i] < donchian_lower_aligned[i]:  # Exit when price falls below lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Donchian bands or opposite signal
            if close[i] > donchian_upper_aligned[i]:  # Exit when price rises above upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0