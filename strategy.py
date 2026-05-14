#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and 1w volume spike confirmation.
# Long when price breaks above Donchian upper band AND 1d close > 1d EMA50 (uptrend) AND 1w volume > 2.0 * 20-period average volume.
# Short when price breaks below Donchian lower band AND 1d close < 1d EMA50 (downtrend) AND 1w volume > 2.0 * 20-period average volume.
# Exit when price retraces to the Donchian midpoint (average of upper and lower bands).
# Uses discrete position sizing (0.30) to limit fee churn. Designed for 4h timeframe with strict entry conditions to avoid overtrading.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_Donchian20_Breakout_1dEMA50_1wVolumeSpike_v1"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1w volume confirmation filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    volume_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1w = volume_1w > (2.0 * vol_ma_20_1w)  # Volume > 2.0x 20-period MA
    volume_confirm_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm_1w.astype(float))
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < lookback - 1:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            window_high = high[i - lookback + 1:i + 1]
            window_low = low[i - lookback + 1:i + 1]
            highest_high[i] = np.max(window_high)
            lowest_low[i] = np.min(window_low)
    
    # Donchian midpoint (exit level)
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm_1w_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = prices.index[i].hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper band AND 1d close > 1d EMA50 (uptrend) AND volume confirmation
            if (low[i] <= highest_high[i] and close[i] > highest_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm_1w_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # SHORT: price breaks below Donchian lower band AND 1d close < 1d EMA50 (downtrend) AND volume confirmation
            elif (high[i] >= lowest_low[i] and close[i] < lowest_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm_1w_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals