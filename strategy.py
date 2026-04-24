#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and Camarilla pivot levels from prior day.
- Entry: Long when price breaks above H3 with volume spike AND price > 1d EMA34.
         Short when price breaks below L3 with volume spike AND price < 1d EMA34.
- Exit: Opposite Camarilla break (price < L3 for long, price > H3 for short) OR EMA34 cross in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels provide institutional support/resistance. Breakouts with volume confirm institutional participation.
- EMA34 filter ensures trading with higher timeframe trend.
- Works in bull markets (buy H3 breakouts in uptrend) and bear markets (sell L3 breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla breakout frequency with filters.
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
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate prior day's Camarilla levels (H3, L3) from 1d data
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # Use prior day's values to avoid look-ahead
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    camarilla_h3 = df_1d_close + 1.1 * (df_1d_high - df_1d_low) / 4
    camarilla_l3 = df_1d_close - 1.1 * (df_1d_high - df_1d_low) / 4
    
    # Align Camarilla levels to 6h timeframe (prior day's levels available after 1d close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3, additional_delay_bars=1)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3, additional_delay_bars=1)
    
    # Volume spike: current volume > 1.5 * 20-period EMA of volume
    volume_ema20 = ema(volume, 20)
    volume_spike = volume > 1.5 * volume_ema20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Exit conditions
        if position != 0:
            if position == 1:  # Long position
                # Exit if price breaks below L3 OR closes below EMA34
                if curr_low < camarilla_l3_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            elif position == -1:  # Short position
                # Exit if price breaks above H3 OR closes above EMA34
                if curr_high > camarilla_h3_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above H3 with volume spike AND bullish trend
            if curr_high > camarilla_h3_aligned[i] and volume_spike[i] and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume spike AND bearish trend
            elif curr_low < camarilla_l3_aligned[i] and volume_spike[i] and curr_close < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Maintain long signal
            signals[i] = 0.25
        elif position == -1:
            # Maintain short signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0