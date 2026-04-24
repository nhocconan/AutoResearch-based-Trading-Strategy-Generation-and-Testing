#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for ATR regime and Donchian levels.
- Donchian channels calculated from previous 1d OHLC (20-period high/low).
- ATR regime: ATR(14) > ATR(50) indicates high volatility trending market.
- Entry: Long when price breaks above 1d Donchian high with volume spike and ATR regime bullish.
         Short when price breaks below 1d Donchian low with volume spike and ATR regime bearish.
- Exit: When price returns to the midpoint of the Donchian channel (mean reversion).
- Works in bull via buying breakouts in high volatility uptrends, in bear via selling breakdowns in high volatility downtrends.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First TR
    tr2[0] = np.nan  # No previous close
    tr3[0] = np.nan  # No previous close
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.full(len(tr), np.nan)
    atr[period-1] = np.nanmean(tr[:period])  # Initial ATR
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian levels and ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    atr_14 = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_50 = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 50)
    
    # ATR regime: True when ATR(14) > ATR(50) (high volatility/trending market)
    atr_regime = atr_14 > atr_50
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 1d indicators to 12h
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # Need enough 1d bars for ATR50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and ATR regime
            if volume_spike[i]:
                # Only trade in high volatility regime
                if atr_regime_aligned[i]:
                    # Bullish breakout: price > Donchian high
                    if close[i] > donchian_high_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakdown: price < Donchian low
                    elif close[i] < donchian_low_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price returns to Donchian midpoint (mean reversion)
            if close[i] <= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian midpoint (mean reversion)
            if close[i] >= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATRRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0