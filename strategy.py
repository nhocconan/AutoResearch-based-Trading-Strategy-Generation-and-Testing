#!/usr/bin/env python3
# Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above 20-period Donchian high AND price > 4h EMA50 (uptrend) AND 1d volume > 2.0x 20-period average.
# Short when price breaks below 20-period Donchian low AND price < 4h EMA50 (downtrend) AND 1d volume > 2.0x 20-period average.
# Exit on opposite Donchian breakout (long exits on low break, short exits on high break).
# Uses 4h HTF for trend to reduce whipsaw, 1d HTF for volume confirmation to avoid low-liquidity noise.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe to stay within fee drag limits.
# Donchian provides clear structure, EMA50 filters counter-trend noise, volume spike confirms participation.

name = "1h_Donchian20_Breakout_4hEMA50_1dVolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1h Indicators (LTF) ---
    # 1h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) - trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: > 2.0x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_confirm_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND uptrend (price > 4h EMA50) AND 1d volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_confirm_1d_aligned[i] > 0.5):  # Treat as boolean (>0.5 = True)
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Donchian low AND downtrend (price < 4h EMA50) AND 1d volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low (contrarian exit)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high (contrarian exit)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals