#!/usr/bin/env python3
# Hypothesis: 6h Williams %R with 12h EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 or > -20) signal potential reversals.
# Long when Williams %R crosses above -80 from below AND price > 12h EMA50 AND volume > 1.5x 20-period average.
# Short when Williams %R crosses below -20 from above AND price < 12h EMA50 AND volume > 1.5x 20-period average.
# Exit on opposite signal or ATR(14) trailing stop (2.5x).
# Uses 6h primary timeframe with 12h trend filter to target 50-150 total trades over 4 years.
# Williams %R is mean-reversion oriented but combined with trend filter to avoid counter-trend trades in strong moves.
# Volume confirmation ensures breakouts have participation. Designed for BTC/ETH with strict entry conditions.

name = "6h_WilliamsR_12hEMA50_VolumeSpike_v1"
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
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    denom = highest_high - lowest_low
    denom = np.where(denom == 0, 1e-10, denom)
    williams_r = -100 * (highest_high - close) / denom
    
    # Get 12h data for EMA50 trend filter (MTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h close
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 6h timeframe (wait for completed 12h bar)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current 6h volume > 1.5x 20-period average (spike confirmation)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below AND price > 12h EMA50 AND volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Williams %R crosses below -20 from above AND price < 12h EMA50 AND volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
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
            # EXIT LONG: trailing stop hit or opposite signal
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            opposite_signal = williams_r[i] < -80 and williams_r[i-1] >= -80  # Re-entry condition
            if trailing_stop or opposite_signal:
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
            # EXIT SHORT: trailing stop hit or opposite signal
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            opposite_signal = williams_r[i] > -20 and williams_r[i-1] <= -20  # Re-entry condition
            if trailing_stop or opposite_signal:
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