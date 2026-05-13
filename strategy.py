#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout + 1w EMA200 trend filter + volume confirmation.
# Long when price breaks above Donchian upper band AND close > 1w EMA200 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower band AND close < 1w EMA200 AND volume > 1.5x 20-period average
# Exit on opposite Donchian breakout or ATR trailing stop (2.5x)
# Uses 1d timeframe with 1w trend filter for noise reduction, targeting 30-100 trades over 4 years.
# Donchian channels provide clear structural breaks, 1w EMA200 filters intermediate trend, volume confirms authenticity.

name = "1d_Donchian20_1wEMA200_VolumeSpike_v1"
timeframe = "1d"
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
    
    # Donchian(20) on 1d: 20-period high/low
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_upper = rolling_max(high, 20)
    donchian_lower = rolling_min(low, 20)
    
    # Get 1w data for EMA200 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on 1w close
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF arrays to 1d timeframe (wait for completed 1w bar)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Volume filter: current 1d volume > 1.5x 20-period average (spike confirmation)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(200, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        upper_breakout = close[i] > donchian_upper[i-1]  # Break above previous upper band
        lower_breakout = close[i] < donchian_lower[i-1]  # Break below previous lower band
        
        if position == 0:
            # LONG: upper breakout AND close > 1w EMA200 AND volume spike
            if upper_breakout and close[i] > ema200_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: lower breakout AND close < 1w EMA200 AND volume spike
            elif lower_breakout and close[i] < ema200_1w_aligned[i] and volume_filter[i]:
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
            # EXIT LONG: lower breakout OR trailing stop hit
            breakout_exit = lower_breakout
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
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
            # EXIT SHORT: upper breakout OR trailing stop hit
            breakout_exit = upper_breakout
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
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