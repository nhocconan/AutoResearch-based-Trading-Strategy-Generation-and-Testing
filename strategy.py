#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ATR filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d ATR(14) for volatility filter (only trade when ATR > 20-period ATR MA to avoid low-volatility chop).
- Volume: Current 4h volume > 1.5 * 20-period volume MA to confirm breakout strength.
- Entry: Long when price breaks above Donchian(20) high AND 1d ATR filter AND volume spike.
         Short when price breaks below Donchian(20) low AND 1d ATR filter AND volume spike.
- Exit: Opposite Donchian(10) level (tighter for profit taking) or loss of volume/ATR filter.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian breakouts capture strong momentum moves, while ATR and volume filters avoid false breakouts in low-volatility environments.
Works in both bull and bear markets by only taking breakouts in the direction of volatility expansion.
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
    
    # Calculate Donchian channels for 4h (20-period for entry, 10-period for exit)
    # Donchian high = max(high over period), Donchian low = min(low over period)
    # Using rolling window with min_periods to avoid look-ahead
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    prev_close = np.roll(df_1d_close, 1)
    prev_close[0] = np.nan
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - prev_close)
    tr3 = np.abs(df_1d_low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR = smoothed TR (using Wilder's smoothing: ATR today = ( ATR yesterday * (n-1) + TR today ) / n)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period ATR MA for filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d volume data for volume confirmation (using 1d volume as proxy for market activity)
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Filters: ATR > ATR MA (volatility expansion) and volume spike
    atr_filter = atr_1d_aligned > atr_ma_1d_aligned
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)  # Using 1d volume MA as benchmark
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20)  # Need enough bars for Donchian20, ATR14, and MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(atr_filter[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with filters
            if atr_filter[i] and volume_spike[i]:
                # Bullish: price breaks above Donchian(20) high
                if curr_high > donchian_high_20[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian(20) low
                elif curr_low < donchian_low_20[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian(10) low OR loss of filters
            if curr_low < donchian_low_10[i] or not (atr_filter[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian(10) high OR loss of filters
            if curr_high > donchian_high_10[i] or not (atr_filter[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRFilter_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0