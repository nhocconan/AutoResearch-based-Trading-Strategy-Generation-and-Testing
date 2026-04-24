#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d volume spike and ATR regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume average and ATR calculation.
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3).
  Long when Lips > Teeth > Jaw (bullish alignment). Short when Lips < Teeth < Jaw (bearish alignment).
- Entry: Long when Alligator bullish AND volume > 2.0 * 1d average volume AND ATR(14) < ATR(50) (low volatility regime).
         Short when Alligator bearish AND volume > 2.0 * 1d average volume AND ATR(14) < ATR(50).
- Exit: Opposite Alligator alignment signal.
- Signal size: 0.25 discrete to minimize fee drag.
- Alligator captures trend emergence after consolidation.
- Volume confirmation ensures breakout legitimacy.
- ATR regime filter avoids high-volatility choppy markets where signals fail.
- Works in both bull and bear markets as it captures volatility expansion after contraction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA) with proper min_periods."""
    series = pd.Series(series)
    sma = series.rolling(window=period, min_periods=period).mean()
    smma_values = np.full(len(series), np.nan)
    smma_values[period-1] = sma.iloc[period-1]
    for i in range(period, len(series)):
        smma_values[i] = (smma_values[i-1] * (period-1) + series.iloc[i]) / period
    return smma_values

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
    
    # Calculate 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d ATR for regime filter
    if len(df_1d) < 50:  # Need sufficient data for ATR(50)
        return np.zeros(n)
    
    atr_14_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_50_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 50)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate Williams Alligator from 12h data
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    jaw = smma(close, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    
    # Apply shifts (shift right = delay)
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > jaw_shift:
        jaw_shifted[jaw_shift:] = jaw[:-jaw_shift]
    if len(teeth) > teeth_shift:
        teeth_shifted[teeth_shift:] = teeth[:-teeth_shift]
    if len(lips) > lips_shift:
        lips_shifted[lips_shift:] = lips[:-lips_shift]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_50_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Alligator alignment
        bullish_aligned = lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]
        bearish_aligned = lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
        volume_confirm = curr_volume > 2.0 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
        
        # ATR regime filter: ATR(14) < ATR(50) (low volatility regime)
        atr_regime = atr_14_1d_aligned[i] < atr_50_1d_aligned[i]
        
        # Exit conditions: opposite Alligator alignment
        if position != 0:
            # Exit long: Alligator turns bearish
            if position == 1 and bearish_aligned:
                signals[i] = 0.0
                position = 0
                continue
            # Exit short: Alligator turns bullish
            elif position == -1 and bullish_aligned:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Alligator alignment with volume confirmation and ATR regime filter
        if position == 0:
            if bullish_aligned and volume_confirm and atr_regime:
                signals[i] = 0.25
                position = 1
            elif bearish_aligned and volume_confirm and atr_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_ATRRegime_v1"
timeframe = "12h"
leverage = 1.0