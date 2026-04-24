#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ATR filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d ATR(14) to filter breakouts by volatility regime (only trade when ATR > 20-period MA).
- Volume: Current 4h volume > 1.5 * 20-period volume MA to ensure participation.
- Entry: Long when price breaks above Donchian upper (20) AND ATR filter bullish AND volume spike.
         Short when price breaks below Donchian lower (20) AND ATR filter bullish AND volume spike.
         (Note: ATR filter is regime-based, not directional - we trade breakouts in high volatility)
- Exit: Opposite Donchian level (lower for long, upper for short) or loss of volume/ATR confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian breakouts capture strong moves, while ATR filter ensures we only trade during sufficient volatility,
avoiding false breakouts in low-volatility chop. Volume confirmation adds participation validation.
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
    
    # Calculate Donchian channels (20-period) on 4h - using previous bar to avoid look-ahead
    # Upper = max(high of last 20 periods), Lower = min(low of last 20 periods)
    # We use rolling window on previous bar's data
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate rolling max/min on previous bar's high/low
    high_series = pd.Series(prev_high)
    low_series = pd.Series(prev_low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr1[0] = np.nan  # First bar has no previous close
    tr2[0] = np.nan
    tr3[0] = np.nan
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period MA of 1d ATR for regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d volume data for volume confirmation (using 1d volume MA as filter)
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Filters: ATR regime (current ATR > 20-period MA) and volume spike
    atr_regime = atr_1d_aligned > atr_ma_1d_aligned
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20)  # Donchian(20), ATR(14), ATR MA(20), Vol MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr_regime[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volatility regime and volume spike
            if atr_regime[i] and volume_spike[i]:
                # Bullish: price breaks above Donchian upper
                if curr_high > donchian_upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian lower
                elif curr_low < donchian_lower[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR loss of confirmation
            if curr_low < donchian_lower[i] or not (atr_regime[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR loss of confirmation
            if curr_high > donchian_upper[i] or not (atr_regime[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_VolumeRegime_v1"
timeframe = "4h"
leverage = 1.0