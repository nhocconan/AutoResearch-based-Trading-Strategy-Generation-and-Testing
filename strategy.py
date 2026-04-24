#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 12h EMA34 trend filter and 6h volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA34 trend filter.
- Entry: Long when Williams %R(14) < -80 (oversold) AND price > 12h EMA34 AND volume > 1.5 * 20-period average volume.
         Short when Williams %R(14) > -20 (overbought) AND price < 12h EMA34 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Williams %R extreme (long exits when %R > -20, short exits when %R < -80) OR price crosses 12h EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R identifies exhaustion points in both bull and bear markets; trend filter ensures trading with the 12h trend.
- Volume spike confirms participation, reducing false signals from low-volume reversals.
- Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Williams %R extremes with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def williams_r(high, low, close, period):
    """Calculate Williams %R."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h, additional_delay_bars=1)
    
    # Calculate 6h Williams %R(14)
    wr = williams_r(high, low, close, 14)
    
    # Calculate 6h volume spike: current volume > 1.5 * 20-period average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / (vol_ma20 + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(wr[i]) or np.isnan(ema34_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Williams %R extreme OR price crosses 12h EMA34 in opposite direction
        if position != 0:
            # Exit long: Williams %R > -20 (overbought) OR price falls below 12h EMA34
            if position == 1:
                if wr[i] > -20 or curr_close < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R < -80 (oversold) OR price rises above 12h EMA34
            elif position == -1:
                if wr[i] < -80 or curr_close > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with trend filter and volume confirmation
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND bullish 12h trend AND volume spike
            if wr[i] < -80 and curr_close > ema34_12h_aligned[i] and volume_spike[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND bearish 12h trend AND volume spike
            elif wr[i] > -20 and curr_close < ema34_12h_aligned[i] and volume_spike[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0