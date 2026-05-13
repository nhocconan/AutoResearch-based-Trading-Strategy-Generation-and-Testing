#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume spike confirmation.
# Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA34 (uptrend) AND volume > 2.0x 20-period average.
# Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA34 (downtrend) AND volume > 2.0x 20-period average.
# Williams %R identifies exhaustion points in 6h cycles, while 1d EMA34 filters for intermediate-term trend alignment.
# Volume spike confirms institutional participation at turning points. Target: 60-120 total trades over 4 years (15-30/year) on 6h.

name = "6h_WilliamsR_MeanReversion_1dEMA34_VolumeSpike_v1"
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
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Oversold Williams %R AND price above 1d EMA34 (uptrend) AND volume spike
            if williams_r[i] < -80 and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought Williams %R AND price below 1d EMA34 (downtrend) AND volume spike
            elif williams_r[i] > -20 and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to neutral territory (> -50) OR volume dries up
            if williams_r[i] > -50 or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to neutral territory (< -50) OR volume dries up
            if williams_r[i] < -50 or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals