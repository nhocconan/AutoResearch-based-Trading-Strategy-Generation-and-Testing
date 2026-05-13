#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 12h EMA34 trend filter and 1d volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND price > 12h EMA34 AND 1d volume > 1.5 * 20-period average volume.
# Short when Williams %R > -20 (overbought) AND price < 12h EMA34 AND 1d volume > 1.5 * 20-period average volume.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing mean reversals in trending markets with volume confirmation.
# Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe.

name = "6h_WilliamsR_Reversal_12hEMA34_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Williams %R (14-period) on primary timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 12h EMA34 AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_spike_aligned[i] > 0.5):  # True if volume spike aligned
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND price < 12h EMA34 AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals