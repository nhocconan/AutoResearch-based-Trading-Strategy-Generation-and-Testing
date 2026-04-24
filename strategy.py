#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla pivot (R1/S1) breakout with 1w volume spike and ATR regime filter.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for Camarilla pivot calculation (based on prior week OHLC), volume average and ATR.
- Camarilla Pivots: identifies key support/resistance levels from prior 1w range.
- Entry: Long when price breaks above R1 AND volume > 1.8 * 10-period average volume AND ATR(10) < ATR(30) (low volatility regime).
         Short when price breaks below S1 AND volume > 1.8 * 10-period average volume AND ATR(10) < ATR(30).
- Exit: Opposite Camarilla breakout (price crosses back below R1 for longs, above S1 for shorts).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets as it captures volatility expansion after contraction.
- Using 1d timeframe reduces trade frequency vs lower timeframes, minimizing fee drag while capturing significant moves.
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
    if n < 40:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Camarilla pivots (R1, S1) from prior 1w OHLC
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 weeks for prior week calculation
        return np.zeros(n)
    
    # Prior week OHLC for Camarilla calculation
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Calculate 1w volume average for confirmation (10-period)
    if len(df_1w) < 10:
        return np.zeros(n)
    
    vol_ma_10 = pd.Series(df_1w['volume'].values).rolling(window=10, min_periods=10).mean().values
    vol_ma_10_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10)
    
    # Calculate 1w ATR for regime filter
    if len(df_1w) < 30:
        return np.zeros(n)
    
    atr_10_1w = atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 10)
    atr_30_1w = atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 30)
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    atr_30_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_30_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(10, 30)  # Need 10 for volume MA, 30 for ATR(30)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_10_aligned[i]) or np.isnan(atr_10_1w_aligned[i]) or
            np.isnan(atr_30_1w_aligned[i])):
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
            
            # Volume confirmation: current volume > 1.8 * 10-period average volume (aligned)
            volume_confirm = curr_volume > 1.8 * vol_ma_10_aligned[i] if not np.isnan(vol_ma_10_aligned[i]) else False
            
            # ATR regime filter: ATR(10) < ATR(30) (low volatility regime)
            atr_regime = atr_10_1w_aligned[i] < atr_30_1w_aligned[i]
            
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

name = "1d_Camarilla_R1S1_Breakout_1wVolumeSpike_ATRRegime_v1"
timeframe = "1d"
leverage = 1.0