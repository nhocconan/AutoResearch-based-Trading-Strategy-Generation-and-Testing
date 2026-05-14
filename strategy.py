#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation (>1.5x 20-period average).
# Long when price breaks above upper Donchian channel AND close > 1w EMA34 (bullish trend) AND volume > 1.5x MA20.
# Short when price breaks below lower Donchian channel AND close < 1w EMA34 (bearish trend) AND volume > 1.5x MA20.
# Exit when price crosses the 20-period EMA in opposite direction or Donchian breakout fails.
# Uses 1w HTF for trend to reduce noise and overtrading. Volume confirmation (>1.5x) reduces false signals.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits for 1d timeframe.
# Donchian channels provide clear structure, EMA34 filters trend direction, volume confirms conviction.

name = "1d_Donchian20_Breakout_1wEMA34_VolumeConfirm_v1"
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
    
    # --- 1d Indicators (LTF) ---
    # Donchian Channel (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # 20-period EMA for exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # 1d volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume > (1.5 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) - trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(volume_confirm_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian AND close > 1w EMA34 (bullish trend) AND volume confirm
            if (close[i] > highest_20[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_confirm_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian AND close < 1w EMA34 (bearish trend) AND volume confirm
            elif (close[i] < lowest_20[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_confirm_1d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 20 EMA OR Donchian breakout fails (price < upper channel)
            if (close[i] < ema_20[i] or 
                close[i] < highest_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 20 EMA OR Donchian breakout fails (price > lower channel)
            if (close[i] > ema_20[i] or 
                close[i] > lowest_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals