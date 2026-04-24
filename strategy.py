#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h volume spike and 12h ATR regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for volume average and ATR calculation (reduces noise vs 1d).
- Donchian Channel: identifies breakouts from 20-period high/low.
- Entry: Long when price breaks above 20-period high AND volume > 1.8 * 12h average volume AND ATR(14) < ATR(50) (low volatility regime).
         Short when price breaks below 20-period low AND volume > 1.8 * 12h average volume AND ATR(14) < ATR(50).
- Exit: Opposite Donchian breakout signal.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakouts capture strong momentum moves.
- Volume confirmation ensures breakout legitimacy.
- ATR regime filter avoids high-volatility choppy markets where breakouts fail.
- Works in both bull and bear markets as it captures volatility expansion after contraction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def atr(high, low, close, period):
    """Calculate Average True Range with proper min_periods."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr_values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h volume average for confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Calculate 12h ATR for regime filter
    if len(df_12h) < 50:  # Need sufficient data for ATR(50)
        return np.zeros(n)
    
    atr_14_12h = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    atr_50_12h = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 50)
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    atr_50_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_50_12h)
    
    # Calculate Donchian Channel from 4h data (20-period)
    donchian_period = 20
    donchian_high = pd.Series(close).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(close).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 20, 50)  # Need 20 for Donchian, 20 for volume MA, 50 for ATR(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or
            np.isnan(atr_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price breaks below Donchian low
            if position == 1:
                if curr_low <= donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high
            elif position == -1:
                if curr_high >= donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and ATR regime filter
        if position == 0:
            # Donchian breakout signals
            breakout_up = curr_high >= donchian_high[i] and prev_close < donchian_high[i-1]
            breakout_down = curr_low <= donchian_low[i] and prev_close > donchian_low[i-1]
            
            # Volume confirmation: current volume > 1.8 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.8 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # ATR regime filter: ATR(14) < ATR(50) (low volatility regime)
            atr_regime = atr_14_12h_aligned[i] < atr_50_12h_aligned[i]
            
            if breakout_up and volume_confirm and atr_regime:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and atr_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hVolumeSpike_ATRRegime_v1"
timeframe = "4h"
leverage = 1.0