#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and 1w EMA200 trend filter.
# Long when price breaks above Donchian upper band AND 1d ATR(14) > 1.5 * 20-period mean ATR (high volatility regime) AND 1w EMA200 > EMA200 5 periods ago (bullish long-term trend).
# Short when price breaks below Donchian lower band AND 1d ATR(14) > 1.5 * 20-period mean ATR AND 1w EMA200 < EMA200 5 periods ago (bearish long-term trend).
# Exit when price retraces to the Donchian middle band (20-period midpoint).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 4h timeframe with strict entry conditions.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h.

name = "4h_Donchian20_Breakout_1dATR_VolFilter_1wEMA200_Trend_v1"
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
    
    # Calculate Donchian channels (20-period) for entry/exit
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_20 + lowest_20) / 2.0
    
    # Calculate 1d ATR(14) for volatility filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr_14 > (1.5 * atr_ma_20)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter.astype(float))
    
    # Calculate 1w EMA200 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_prev = np.roll(ema_200, 5)
    ema_200_prev[:5] = np.nan
    ema_trend = ema_200 > ema_200_prev  # Rising EMA200 = bullish trend
    ema_trend_aligned = align_htf_to_ltf(prices, df_1w, ema_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(middle_20[i]) or
            np.isnan(vol_filter_aligned[i]) or np.isnan(ema_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper band AND high volatility regime AND bullish long-term trend
            if (open_[i] <= highest_20[i] and close[i] > highest_20[i] and 
                vol_filter_aligned[i] > 0.5 and 
                ema_trend_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower band AND high volatility regime AND bearish long-term trend
            elif (open_[i] >= lowest_20[i] and close[i] < lowest_20[i] and 
                  vol_filter_aligned[i] > 0.5 and 
                  ema_trend_aligned[i] < 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian middle band
            if close[i] <= middle_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian middle band
            if close[i] >= middle_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals