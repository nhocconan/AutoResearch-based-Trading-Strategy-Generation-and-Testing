#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and 1d volume spike confirmation.
# Long when price breaks above Donchian upper band with 1d EMA34 uptrend and 1d volume > 2.0x 20-period average.
# Short when price breaks below Donchian lower band with 1d EMA34 downtrend and 1d volume > 2.0x 20-period average.
# Exit on opposite Donchian band (lower for longs, upper for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn and strict volume confirmation to reduce false breakouts.
# Target: 100-200 total trades over 4 years = 25-50/year for 4h timeframe.
# Works in bull/bear: 1d EMA34 ensures strong trend alignment, Donchian(20) provides clear structure, 1d volume spike confirms institutional participation.

name = "4h_Donchian20_Breakout_1dEMA34_1dVolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # 4h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d EMA34 uptrend/downtrend signals
    ema_34_uptrend = ema_34_1d_aligned > np.roll(ema_34_1d_aligned, 1)
    ema_34_downtrend = ema_34_1d_aligned < np.roll(ema_34_1d_aligned, 1)
    # Handle first value
    ema_34_uptrend[0] = False
    ema_34_downtrend[0] = False
    
    # 1d volume confirmation: > 2.0x 20-period average (volume spike)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    
    # Align 1d volume confirmation to 4h
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(ema_34_uptrend[i]) or 
            np.isnan(ema_34_downtrend[i]) or
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band + 1d EMA34 uptrend + 1d volume spike
            if (close[i] > highest_20[i] and 
                ema_34_uptrend[i] and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band + 1d EMA34 downtrend + 1d volume spike
            elif (close[i] < lowest_20[i] and 
                  ema_34_downtrend[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band
            if close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band
            if close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals