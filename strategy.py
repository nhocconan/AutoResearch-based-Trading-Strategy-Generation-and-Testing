#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and 1d volume confirmation.
# Long when price breaks above 20-day high with 1w EMA50 uptrend and 1d volume > 1.5x 20-period average.
# Short when price breaks below 20-day low with 1w EMA50 downtrend and 1d volume > 1.5x 20-period average.
# Exit on opposite Donchian level (20-day low for longs, 20-day high for shorts).
# Uses discrete position sizing (0.30) to balance return and drawdown. Target: 30-100 trades over 4 years.
# Works in bull/bear: 1w EMA50 ensures strong trend alignment, Donchian provides clear breakout levels, volume confirms conviction.

name = "1d_Donchian20_Breakout_1wEMA50_1dVolumeConfirm"
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
    
    # --- 1d Indicators ---
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA50 trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w EMA50 uptrend/downtrend (current > previous)
    ema_50_uptrend = ema_50_1w_aligned > np.roll(ema_50_1w_aligned, 1)
    ema_50_downtrend = ema_50_1w_aligned < np.roll(ema_50_1w_aligned, 1)
    # Handle first value
    ema_50_uptrend[0] = False
    ema_50_downtrend[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_uptrend[i]) or 
            np.isnan(ema_50_downtrend[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + 1w EMA50 uptrend + volume confirmation
            if (close[i] > donchian_high[i] and 
                ema_50_uptrend[i] and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian low + 1w EMA50 downtrend + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema_50_downtrend[i] and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals