#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper channel with 1d EMA34 uptrend and 4h volume > 1.8x 20-period average.
# Short when price breaks below 4h Donchian lower channel with 1d EMA34 downtrend and 4h volume > 1.8x 20-period average.
# Exit on opposite Donchian channel touch or at 1d close level.
# Uses discrete position sizing (0.25) to limit fee churn and strict volume confirmation to reduce false breakouts.
# Target: 75-200 trades over 4 years (19-50/year) for 4h timeframe.
# Works in bull/bear: 1d EMA34 ensures trend alignment, Donchian provides structure within trend.

name = "4h_Donchian20_Breakout_1dEMA34_Trend_Volume"
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
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # 4h Volume confirmation: > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_bullish = close_1d > ema_34  # Bullish if price above EMA34
    ema_34_bearish = close_1d < ema_34  # Bearish if price below EMA34
    
    # Align 1d indicators to 4h
    ema_34_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_34_bullish.astype(float))
    ema_34_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_34_bearish.astype(float))
    
    # 1d close for exit level (prior day close)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_bullish_aligned[i]) or 
            np.isnan(ema_34_bearish_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + 1d uptrend + volume confirmation
            if (close[i] > donchian_upper[i] and 
                ema_34_bullish_aligned[i] > 0.5 and 
                volume_confirm[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + 1d downtrend + volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  ema_34_bearish_aligned[i] > 0.5 and 
                  volume_confirm[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches Donchian lower OR price <= 1d close
            if close[i] <= donchian_lower[i] or close[i] <= close_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches Donchian upper OR price >= 1d close
            if close[i] >= donchian_upper[i] or close[i] >= close_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals