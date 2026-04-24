#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and Camarilla pivot calculation.
- Entry: Long when price breaks above H3 with volume > 1.5x 20-period MA AND price > 1d EMA34.
         Short when price breaks below L3 with volume > 1.5x 20-period MA AND price < 1d EMA34.
- Exit: Price retouches the 1d EMA34 level or opposite Camarilla breakout occurs.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide mathematically derived support/resistance from prior day.
- Volume confirmation ensures breakout validity.
- 1d EMA34 filter ensures trading with the higher timeframe trend.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HTF data for Camarilla pivots and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    camarilla_h3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_l3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed for pivot points)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period SMA
    vol_ma20 = sma(volume, 20)
    volume_confirm = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: price retouches 1d EMA34 or opposite Camarilla breakout
        if position != 0:
            # Exit long: price falls below 1d EMA34 OR breaks below L3 (opposite breakdown)
            if position == 1:
                if curr_close < ema34_1d_aligned[i] or curr_close < camarilla_l3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above 1d EMA34 OR breaks above H3 (opposite breakout)
            elif position == -1:
                if curr_close > ema34_1d_aligned[i] or curr_close > camarilla_h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above H3 with volume confirmation AND bullish 1d trend
            if (curr_close > camarilla_h3_aligned[i] and 
                volume_confirm[i] and 
                curr_close > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume confirmation AND bearish 1d trend
            elif (curr_close < camarilla_l3_aligned[i] and 
                  volume_confirm[i] and 
                  curr_close < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0