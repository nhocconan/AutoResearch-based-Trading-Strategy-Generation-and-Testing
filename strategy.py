#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above 20-day high with 1w EMA34 uptrend and volume > 1.5x 20-day average.
# Short when price breaks below 20-day low with 1w EMA34 downtrend and volume > 1.5x 20-day average.
# Exit on opposite Donchian level or at prior day close.
# Uses 08-20 UTC session filter. Position size 0.25 to balance capture and fee drag.
# Target: 50-100 trades over 4 years (12-25/year) for 1d timeframe.
# Works in bull/bear: 1w EMA34 ensures major trend alignment, Donchian provides structure.
# Uses discrete sizing and strict volume to minimize churn.

name = "1d_Donchian20_Breakout_1wEMA34_Trend_Volume_Session"
timeframe = "1d"
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
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # --- 1d Indicators ---
    # Volume confirmation: > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Donchian channels (20-day)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_bullish = close_1w > ema_34  # Bullish if price above EMA34
    ema_34_bearish = close_1w < ema_34  # Bearish if price below EMA34
    
    # Align 1w indicators to 1d
    ema_34_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_34_bullish.astype(float))
    ema_34_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_34_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if outside session or missing data
        if (not in_session[i] or
            np.isnan(ema_34_bullish_aligned[i]) or 
            np.isnan(ema_34_bearish_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + 1w uptrend + volume confirmation
            if (close[i] > donchian_high[i] and 
                ema_34_bullish_aligned[i] > 0.5 and 
                volume_confirm[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + 1w downtrend + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema_34_bearish_aligned[i] > 0.5 and 
                  volume_confirm[i] > 0.5):
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