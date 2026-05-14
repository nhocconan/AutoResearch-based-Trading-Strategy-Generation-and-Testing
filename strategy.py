#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above 20-period high with 1d EMA50 uptrend and 1d volume > 2.0x 20-period average.
# Short when price breaks below 20-period low with 1d EMA50 downtrend and 1d volume > 2.0x 20-period average.
# Exit on opposite Donchian level (20-period low for longs, 20-period high for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn and strict volume confirmation to reduce false breakouts.
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe.
# Works in bull/bear: 1d EMA50 ensures strong trend alignment, Donchian provides clear structure, 1d volume spike confirms institutional participation.

name = "12h_Donchian20_Breakout_1dEMA50_1dVolumeSpike"
timeframe = "12h"
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
    
    # --- 12h Indicators (LTF) ---
    # 12h Donchian channels
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d EMA50 uptrend/downtrend signals
    ema_50_uptrend = ema_50_1d_aligned > np.roll(ema_50_1d_aligned, 1)
    ema_50_downtrend = ema_50_1d_aligned < np.roll(ema_50_1d_aligned, 1)
    # Handle first value
    ema_50_uptrend[0] = False
    ema_50_downtrend[0] = False
    
    # 1d volume confirmation: > 2.0x 20-period average (volume spike)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    
    # Align 1d volume confirmation to 12h
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_uptrend[i]) or 
            np.isnan(ema_50_downtrend[i]) or
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(high_ma_20[i]) or
            np.isnan(low_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-period high + 1d EMA50 uptrend + 1d volume spike
            if (close[i] > high_ma_20[i] and 
                ema_50_uptrend[i] and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low + 1d EMA50 downtrend + 1d volume spike
            elif (close[i] < low_ma_20[i] and 
                  ema_50_downtrend[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low
            if close[i] < low_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high
            if close[i] > high_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals