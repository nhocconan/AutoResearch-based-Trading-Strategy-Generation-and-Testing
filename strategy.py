#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout + 1d EMA34 trend filter + volume spike.
- Primary timeframe: 4h targeting 100-180 total trades over 4 years (25-45/year).
- HTF: 1d for EMA34 trend filter (price above/below EMA34) and volume spike confirmation.
- Entry: Long when price breaks above Camarilla R1 AND close > 1d EMA34 AND volume > 1.5 * 20-period average volume.
         Short when price breaks below Camarilla S1 AND close < 1d EMA34 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Camarilla breakout (price crosses S1 for long, R1 for short) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide intraday support/resistance based on prior day's range.
- EMA34 filter ensures we trade with the higher timeframe trend.
- Volume spike confirms institutional participation and reduces false breakouts.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~140 total over 4 years (~35/year) based on Camarilla breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 20-period average volume for spike detection (on 4h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior 1d bar (OHLC)
    # Camarilla levels are calculated using prior day's high, low, close
    # We need to align the prior day's data to current 4h bars
    df_1d_for_camarilla = get_htf_data(prices, '1d')
    if len(df_1d_for_camarilla) < 2:
        return np.zeros(n)
    
    # Extract prior day's OHLC (shifted by 1 to avoid look-ahead)
    prior_high = df_1d_for_camarilla['high'].shift(1).values
    prior_low = df_1d_for_camarilla['low'].shift(1).values
    prior_close = df_1d_for_camarilla['close'].shift(1).values
    
    # Calculate Camarilla levels
    R1 = prior_close + (prior_high - prior_low) * 1.1 / 12
    S1 = prior_close - (prior_high - prior_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (prior day's levels are constant throughout current day)
    R1_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume spike condition: current volume > 1.5 * 20-period average volume
        volume_spike = curr_volume > 1.5 * vol_ma20[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: price crosses below S1 OR price falls below 1d EMA34
            if position == 1:
                if curr_close < S1_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above R1 OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > R1_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout + trend filter + volume spike
        if position == 0:
            # Long: price breaks above R1 AND above 1d EMA34 AND volume spike
            if curr_close > R1_aligned[i] and curr_close > ema34_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND below 1d EMA34 AND volume spike
            elif curr_close < S1_aligned[i] and curr_close < ema34_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0