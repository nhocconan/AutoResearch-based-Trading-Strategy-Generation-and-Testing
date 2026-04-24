#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 100-180 total trades over 4 years (25-45/year).
- HTF: 1d for EMA34 trend filter and Camarilla pivot levels from prior day.
- Entry: Long when price breaks above R1 with volume spike AND close > 1d EMA34.
         Short when price breaks below S1 with volume spike AND close < 1d EMA34.
- Exit: Price returns to Camarilla pivot point (PP) or opposite breakout occurs.
- Signal size: 0.25 discrete to balance profit potential and fee drag.
- Camarilla levels provide intraday support/resistance based on prior day's range.
- Volume spike confirms institutional participation in breakout.
- Trend filter ensures trades align with higher timeframe momentum.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~140 total over 4 years (~35/year) based on breakout frequency with filters.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate prior day's Camarilla levels for today
    # Need to shift prior day's OHLC to avoid look-ahead
    df_1d_shifted = df_1d.copy()
    df_1d_shifted['high'] = df_1d_shifted['high'].shift(1)
    df_1d_shifted['low'] = df_1d_shifted['low'].shift(1)
    df_1d_shifted['close'] = df_1d_shifted['close'].shift(1)
    
    # Calculate Camarilla levels from prior day
    prior_high = df_1d_shifted['high'].values
    prior_low = df_1d_shifted['low'].values
    prior_close = df_1d_shifted['close'].values
    
    # Camarilla formulas
    PP = (prior_high + prior_low + prior_close) / 3
    R1 = PP + (prior_high - prior_low) * 1.1 / 12
    S1 = PP - (prior_high - prior_low) * 1.1 / 12
    R4 = PP + (prior_high - prior_low) * 1.1 / 2
    S4 = PP - (prior_high - prior_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(PP_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Exit conditions: price returns to pivot point or opposite breakout
        if position != 0:
            # Exit long: price falls back to pivot point OR breaks below S1
            if position == 1:
                if curr_low <= PP_aligned[i] or curr_close < S1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises back to pivot point OR breaks above R1
            elif position == -1:
                if curr_high >= PP_aligned[i] or curr_close > R1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: breakout with volume spike and trend alignment
        if position == 0:
            # Long: price breaks above R1 with volume spike AND bullish trend
            if curr_high > R1_aligned[i] and volume_spike[i] and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike AND bearish trend
            elif curr_low < S1_aligned[i] and volume_spike[i] and curr_close < ema34_1d_aligned[i]:
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