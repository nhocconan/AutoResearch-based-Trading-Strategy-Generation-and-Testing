#!/usr/bin/env python3
# Hypothesis: 4h Williams %R extreme reversal with 1d EMA34 trend filter and 4h volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND price > 1d EMA34 (bullish trend) AND 4h volume > 2.5x 20-period average.
# Short when Williams %R > -20 (overbought) AND price < 1d EMA34 (bearish trend) AND 4h volume > 2.5x 20-period average.
# Exit on opposite Williams %R condition (Williams %R > -50 for longs, < -50 for shorts).
# Uses 1d HTF for trend to reduce noise and overtrading vs shorter trends. Volume spike (2.5x) reduces false signals.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits for 4h timeframe.
# Williams %R captures momentum reversals, effective in both bull and bear markets when combined with HTF trend filter.

name = "4h_WilliamsR_Extreme_1dEMA34_4hVolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # 4h volume confirmation: > 2.5x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (2.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter (smooth for 4h trading)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 1d EMA34 (bullish) AND 4h volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND price < 1d EMA34 (bearish) AND 4h volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (momentum weakening)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (momentum weakening)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals