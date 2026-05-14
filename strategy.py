#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and 1d volume confirmation.
# Long when price breaks above upper Donchian channel AND 1w EMA50 > price (bullish trend) AND 1d volume > 1.5 * 20-period average volume.
# Short when price breaks below lower Donchian channel AND 1w EMA50 < price (bearish trend) AND 1d volume > 1.5 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.30) to limit fee churn. Designed for 1d timeframe with strict entry conditions.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d.

name = "1d_Donchian20_Breakout_1wEMA50_Trend_1dVolumeConfirm_v1"
timeframe = "1d"
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 1d volume confirmation filter (HTF)
    volume_1d = df_1w['volume'].values if 'volume' in df_1w.columns else np.zeros(len(df_1w))
    if len(volume_1d) == 0:
        volume_1d = get_htf_data(prices, '1d')['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm_1d.astype(float)) if len(df_1w) == len(volume_1d) else align_htf_to_ltf(prices, get_htf_data(prices, '1d'), volume_confirm_1d.astype(float))
    
    # Calculate Donchian channels (20-period) for 1d timeframe
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    mid_channel = np.full(n, np.nan)
    
    for i in range(n):
        if i < 19:
            upper_channel[i] = np.nan
            lower_channel[i] = np.nan
            mid_channel[i] = np.nan
        else:
            lookback_high = np.max(high[i-19:i+1])
            lookback_low = np.min(low[i-19:i+1])
            upper_channel[i] = lookback_high
            lower_channel[i] = lookback_low
            mid_channel[i] = (lookback_high + lookback_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or
            np.isnan(mid_channel[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian AND 1w EMA50 > price (bullish trend) AND volume confirmation
            if (open_[i] <= upper_channel[i] and close[i] > upper_channel[i] and 
                ema_50_aligned[i] > close[i] and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # SHORT: price breaks below lower Donchian AND 1w EMA50 < price (bearish trend) AND volume confirmation
            elif (open_[i] >= lower_channel[i] and close[i] < lower_channel[i] and 
                  ema_50_aligned[i] < close[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to midpoint of Donchian channel
            if close[i] <= mid_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price retraces to midpoint of Donchian channel
            if close[i] >= mid_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals