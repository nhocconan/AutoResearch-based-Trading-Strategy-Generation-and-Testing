#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d volume spike confirmation and 1w EMA50 trend filter.
# Long when price breaks above Donchian upper band AND 1d volume > 2.0 * 20-period average volume AND close > 1w EMA50.
# Short when price breaks below Donchian lower band AND 1d volume > 2.0 * 20-period average volume AND close < 1w EMA50.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 4h timeframe with strict entry conditions.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h.

name = "4h_Donchian20_Breakout_1dVolumeSpike_1wEMA50_Trend_v1"
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
    
    # Calculate 1d volume confirmation filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate 1w EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
        donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper band AND volume confirmation AND 1w EMA50 uptrend
            if (open_[i] <= donchian_upper[i] and close[i] > donchian_upper[i] and 
                volume_confirm_1d_aligned[i] > 0.5 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower band AND volume confirmation AND 1w EMA50 downtrend
            elif (open_[i] >= donchian_lower[i] and close[i] < donchian_lower[i] and 
                  volume_confirm_1d_aligned[i] > 0.5 and 
                  close[i] < ema_50_1w_aligned[i]):
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