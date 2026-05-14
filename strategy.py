#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above 20-day high with 1w EMA34 uptrend and 1d volume > 1.5x 20-day average.
# Short when price breaks below 20-day low with 1w EMA34 downtrend and 1d volume > 1.5x 20-day average.
# Exit on opposite Donchian level (20-day low for longs, 20-day high for shorts).
# Uses discrete position sizing (0.25) to balance return and drawdown.
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe.
# Works in bull/bear: 1w EMA34 provides adaptive trend filter, Donchian breakouts capture momentum,
# volume confirmation reduces false signals. Designed for low trade frequency to minimize fee drag.

name = "1d_Donchian20_Breakout_1wEMA34_Trend_Volume"
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
    # 1d volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Trend: price above/below EMA34
    ema_uptrend = ema_34_1w_aligned > 0  # placeholder for boolean, will be set below
    ema_downtrend = ema_34_1w_aligned > 0  # placeholder
    
    # Need to compute trend after alignment
    close_series = pd.Series(close)
    ema_uptrend = (close_series > ema_34_1w_aligned).values
    ema_downtrend = (close_series < ema_34_1w_aligned).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + 1w EMA34 uptrend + volume confirmation
            if (close[i] > donchian_high[i] and 
                ema_uptrend[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + 1w EMA34 downtrend + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema_downtrend[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals