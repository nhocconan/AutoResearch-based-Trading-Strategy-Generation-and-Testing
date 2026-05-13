#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume > 2.0x 20-period average.
# Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume > 2.0x 20-period average.
# Exit on opposite Williams %R extreme (long exit at %R > -20, short exit at %R < -80).
# Designed for 6h timeframe to capture mean reversion in ranging markets while respecting higher timeframe trend.
# Uses volume confirmation to avoid false reversals in low momentum environments.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.

name = "6h_WilliamsR_MeanReversion_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r[highest_high == lowest_low] = -50.0
    
    # Get 1d data for EMA50 trend filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 6h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 6h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume spike
            if williams_r[i] < -80.0 and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume spike
            elif williams_r[i] > -20.0 and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -20 (exits oversold territory)
            if williams_r[i] > -20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -80 (exits overbought territory)
            if williams_r[i] < -80.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals