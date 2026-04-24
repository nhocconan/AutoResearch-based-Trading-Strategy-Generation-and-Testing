#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter.
- Entry: Long when price breaks above R3 with volume spike AND close > 1d EMA34.
         Short when price breaks below S3 with volume spike AND close < 1d EMA34.
- Exit: Opposite Camarilla breakout (price crosses H3/L3) OR trend filter reversal.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide institutional support/resistance with high probability reversal/continuation.
- Volume spike confirms institutional participation.
- Trend filter ensures we trade with higher timeframe momentum.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on strict breakout conditions.
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
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar)
    # We need to shift by 1 to avoid look-ahead (use previous bar's HLC)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no previous
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R3 = pivot + (range_hl * 1.1 / 4)
    S3 = pivot - (range_hl * 1.1 / 4)
    H3 = pivot + (range_hl * 1.1 / 2)
    L3 = pivot - (range_hl * 1.1 / 2)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for volume MA and Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Exit conditions: opposite Camarilla breakout OR trend filter reversal
        if position != 0:
            # Exit long: price crosses below H3 OR trend turns bearish
            if position == 1:
                if curr_low < H3[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above L3 OR trend turns bullish
            elif position == -1:
                if curr_high > L3[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume and trend alignment
        if position == 0:
            # Long: price breaks above R3 with volume spike AND bullish trend
            if curr_high > R3[i] and volume_spike[i] and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike AND bearish trend
            elif curr_low < S3[i] and volume_spike[i] and curr_close < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0