#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and 4h volume spike confirmation.
# Long when price breaks above Donchian upper band AND 12h HMA21 is rising AND 4h volume > 1.5 * 20-period average volume.
# Short when price breaks below Donchian lower band AND 12h HMA21 is falling AND 4h volume > 1.5 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 75-200 total trades over 4 years (19-50/year) for 4h.
# Works in both bull and bear markets: 12h HMA filter ensures we only trade with the intermediate-term trend,
# while volume confirmation avoids breakouts in low-participation environments.

name = "4h_Donchian20_Breakout_12hHMA21_Trend_4hVolumeConfirm_v1"
timeframe = "4h"
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
    
    # Calculate 12h HMA21 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    # HMA calculation: WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    wma_half = np.array([wma(close_12h[i:i+half_len], half_len) if i+half_len <= len(close_12h) else np.nan for i in range(len(close_12h))])
    wma_full = np.array([wma(close_12h[i:i+21], 21) if i+21 <= len(close_12h) else np.nan for i in range(len(close_12h))])
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(raw_hma) else np.nan for i in range(len(raw_hma))])
    hma_rising_12h = np.zeros_like(close_12h, dtype=bool)
    hma_rising_12h[1:] = hma_21_12h[1:] > hma_21_12h[:-1]
    hma_rising_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_rising_12h.astype(float))
    
    # Calculate 4h volume confirmation filter (primary TF)
    vol_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (1.5 * vol_ma_20_4h)
    
    # Calculate Donchian channel (20-period) on primary TF
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(hma_rising_12h_aligned[i]) or 
            np.isnan(volume_confirm_4h[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper band AND 12h HMA21 rising AND volume confirmation
            if (open_[i] <= highest_high_20[i] and close[i] > highest_high_20[i] and 
                hma_rising_12h_aligned[i] > 0.5 and 
                volume_confirm_4h[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower band AND 12h HMA21 falling AND volume confirmation
            elif (open_[i] >= lowest_low_20[i] and close[i] < lowest_low_20[i] and 
                  hma_rising_12h_aligned[i] < 0.5 and 
                  volume_confirm_4h[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals