#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (price > SMA50) and volume confirmation (volume > 1.5x 20-period average).
# Long when price breaks above 20-period high AND close > 1d SMA50 AND volume spike.
# Short when price breaks below 20-period low AND close < 1d SMA50 AND volume spike.
# Exit on opposite breakout or ATR trailing stop (2.0x).
# Uses 6h timeframe with 1d trend filter for noise reduction, targeting 50-150 trades over 4 years.
# Donchian channels provide clear breakout levels, 1d SMA50 filters intermediate trend, volume confirms authenticity.

name = "6h_Donchian20_1dSMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) for 6h: based on previous 20 bars to avoid look-ahead
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    # For first 20 bars, use expanding window
    donchian_upper[:20] = high_series.expanding().max().shift(1).values[:20]
    donchian_lower[:20] = low_series.expanding().min().shift(1).values[:20]
    
    # Get 1d data for SMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate SMA50 on 1d close
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF arrays to 6h timeframe (wait for completed 1d bar)
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Volume filter: current 6h volume > 1.5x 20-period average (spike confirmation)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(sma50_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper AND close > 1d SMA50 AND volume spike
            if close[i] > donchian_upper[i] and close[i] > sma50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: price breaks below Donchian lower AND close < 1d SMA50 AND volume spike
            elif close[i] < donchian_lower[i] and close[i] < sma50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: price breaks below Donchian lower (opposite breakout) OR trailing stop hit
            breakout_exit = close[i] < donchian_lower[i]
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if breakout_exit or trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: price breaks above Donchian upper (opposite breakout) OR trailing stop hit
            breakout_exit = close[i] > donchian_upper[i]
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if breakout_exit or trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals