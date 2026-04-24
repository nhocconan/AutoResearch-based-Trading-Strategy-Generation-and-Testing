#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian breakout with 1d ATR filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for ATR and trend context.
- Donchian channel (20-period) breakout: long on upper band break, short on lower band break.
- ATR filter: only trade when 1d ATR(14) > 20-period SMA of ATR (ensures sufficient volatility).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Stoploss: exit when price retraces to midpoint of Donchian channel.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period SMA of ATR for filter
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_filter = atr_14 > atr_ma_20  # only trade when volatility is above average
    
    # Align ATR filter to 12h
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Donchian channel (20-period) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 20)  # ATR filter + Donchian + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(atr_filter_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout with volume and ATR filter
            if atr_filter_aligned[i] and volume_spike[i]:
                if close[i] > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long: hold until price retrace to midpoint
            signals[i] = 0.25
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold until price retrace to midpoint
            signals[i] = -0.25
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_1dATR_Filter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0