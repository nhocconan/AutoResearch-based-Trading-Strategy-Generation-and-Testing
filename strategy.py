#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w volume spike and ATR regime filter.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for volume average and ATR calculation.
- Donchian(20): upper/lower bands from 20-period high/low.
- Entry: Long when price breaks above upper band AND volume > 2.0 * 20-period average volume AND ATR(14) < ATR(50) (low volatility regime).
         Short when price breaks below lower band AND volume > 2.0 * 20-period average volume AND ATR(14) < ATR(50).
- Exit: Opposite Donchian breakout (price crosses back below upper band for longs, above lower band for shorts).
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakouts capture strong momentum moves after consolidation.
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
    
    # Calculate 1w Donchian bands (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 periods for Donchian
        return np.zeros(n)
    
    # Donchian upper and lower bands (20-period high/low)
    donchian_upper = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Calculate 1w volume average for confirmation (20-period)
    if len(df_1w) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    # Calculate 1w ATR for regime filter
    if len(df_1w) < 50:
        return np.zeros(n)
    
    atr_14_1w = atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    atr_50_1w = atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 50)
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    atr_50_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume MA, 50 for ATR(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or
            np.isnan(atr_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: price crosses back below upper band for longs, above lower band for shorts
        if position != 0:
            # Exit long: price crosses below upper band
            if position == 1:
                if curr_close < donchian_upper_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above lower band
            elif position == -1:
                if curr_close > donchian_lower_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and ATR regime filter
        if position == 0:
            # Donchian breakout signals
            breakout_up = curr_high >= donchian_upper_aligned[i] and prev_close < donchian_upper_aligned[i-1]
            breakout_down = curr_low <= donchian_lower_aligned[i] and prev_close > donchian_lower_aligned[i-1]
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 2.0 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # ATR regime filter: ATR(14) < ATR(50) (low volatility regime)
            atr_regime = atr_14_1w_aligned[i] < atr_50_1w_aligned[i]
            
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

name = "1d_Donchian20_Breakout_1wVolumeSpike_ATRRegime_v1"
timeframe = "1d"
leverage = 1.0