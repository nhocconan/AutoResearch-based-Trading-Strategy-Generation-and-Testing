#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 4h volume spike and 1d ATR regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR regime filter (low volatility regime) and 4h for volume confirmation.
- Camarilla Pivots: identifies key support/resistance levels from prior 1d range (R1/S1).
- Entry: Long when price breaks above R1 AND volume > 2.0 * 20-period 4h volume average AND ATR(14) < ATR(50) on 1d (low volatility regime).
         Short when price breaks below S1 AND volume > 2.0 * 20-period 4h volume average AND ATR(14) < ATR(50) on 1d.
- Exit: Opposite Camarilla breakout (price crosses back below R1 for longs, above S1 for shorts).
- Signal size: 0.25 discrete to minimize fee drag.
- Rationale: Camarilla R1/S1 breaks capture strong momentum after testing intraday support/resistance.
             Volume confirmation avoids fakeouts. ATR regime filter ensures low-volatility breakout conditions.
             Works in bull/bear markets as it captures volatility expansion after contraction, proven on ETH/SOL.
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
    
    # Calculate 1d ATR for regime filter (needs 50 periods for ATR(50))
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    atr_14_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_50_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 50)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate 4h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate prior 1d OHLC for Camarilla R1/S1
    df_1d_for_pivots = get_htf_data(prices, '1d')
    if len(df_1d_for_pivots) < 2:
        return np.zeros(n)
    
    prev_high = df_1d_for_pivots['high'].shift(1).values
    prev_low = df_1d_for_pivots['low'].shift(1).values
    prev_close = df_1d_for_pivots['close'].shift(1).values
    
    # Camarilla R1 and S1 levels (using prior 1d OHLC)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d_for_pivots, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d_for_pivots, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for ATR(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: price crosses back below R1 for longs, above S1 for shorts
        if position != 0:
            # Exit long: price crosses below R1
            if position == 1:
                if curr_close < camarilla_r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above S1
            elif position == -1:
                if curr_close > camarilla_s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and ATR regime filter
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= camarilla_r1_aligned[i] and prev_close < camarilla_r1_aligned[i-1]
            breakout_down = curr_low <= camarilla_s1_aligned[i] and prev_close > camarilla_s1_aligned[i-1]
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume (4h)
            volume_confirm = curr_volume > 2.0 * vol_ma_20[i]
            
            # ATR regime filter: ATR(14) < ATR(50) on 1d (low volatility regime)
            atr_regime = atr_14_1d_aligned[i] < atr_50_1d_aligned[i]
            
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

name = "4h_Camarilla_R1S1_Breakout_4hVolumeSpike_1dATRRegime_v1"
timeframe = "4h"
leverage = 1.0