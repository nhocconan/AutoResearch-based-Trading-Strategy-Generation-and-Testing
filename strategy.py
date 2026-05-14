#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme + 1d EMA50 trend + 6h volume spike confirmation.
# Long when Williams %R(14) < -80 (oversold) + price > 1d EMA50 (bullish trend) + 6h volume > 2.0x 20-period average.
# Short when Williams %R(14) > -20 (overbought) + price < 1d EMA50 (bearish trend) + 6h volume > 2.0x 20-period average.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Williams %R captures mean reversion in extreme zones, EMA50 filters trend direction, volume spike confirms momentum.
# Designed for 6h timeframe to balance trade frequency and signal quality, targeting 12-37 trades/year.

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeConfirm"
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
    
    # --- 6h Indicators (LTF) ---
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 6h volume confirmation: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) + price > 1d EMA50 + volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) + price < 1d EMA50 + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (recovering from oversold)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (declining from overbought)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals