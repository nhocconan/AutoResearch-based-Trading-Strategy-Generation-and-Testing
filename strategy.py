#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and 4h volume confirmation.
# Long when price breaks above Donchian upper band with 12h EMA50 > price (uptrend) and 4h volume > 2.0x 20-period average.
# Short when price breaks below Donchian lower band with 12h EMA50 < price (downtrend) and 4h volume > 2.0x 20-period average.
# Exit on opposite Donchian band. Uses discrete position sizing (0.25) to minimize fee churn.
# Volume filter >2.0x reduces false breakouts. 12h EMA50 ensures trend alignment, reducing whipsaws in ranging markets.
# Target: 50-150 total trades over 4 years = 12-37/year for 4h. Works in bull/bear via 12h EMA50 trend filter.

name = "4h_Donchian20_Breakout_12hEMA50_4hVolumeConfirm"
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
    # Donchian channels (20-period)
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (2.0 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(donchian_h[i]) or
            np.isnan(donchian_l[i]) or
            np.isnan(volume_confirm_4h[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + 12h EMA50 > price (uptrend) + 4h volume confirmation
            if (close[i] > donchian_h[i] and 
                ema_50_12h_aligned[i] > close[i] and 
                volume_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + 12h EMA50 < price (downtrend) + 4h volume confirmation
            elif (close[i] < donchian_l[i] and 
                  ema_50_12h_aligned[i] < close[i] and 
                  volume_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band
            if close[i] < donchian_l[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band
            if close[i] > donchian_h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals