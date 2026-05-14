#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation (>1.8x 20-period average).
# Long when price breaks above upper Donchian band AND close > 1w EMA34 AND volume > 1.8x MA20.
# Short when price breaks below lower Donchian band AND close < 1w EMA34 AND volume > 1.8x MA20.
# Exit when price crosses the 1w EMA34 in opposite direction.
# Uses 1w HTF for primary trend to reduce noise and overtrading. Higher volume threshold reduces false signals.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
# Donchian breakouts capture strong momentum moves, while 1w EMA filter ensures alignment with weekly trend.

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
    # Volume confirmation: > 1.8x 20-period average (higher threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
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
    
    for i in range(20, n):  # Need 20 bars for Donchian calculation
        # Skip if missing data
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(high[i-20:i].max()) or
            np.isnan(low[i-20:i].min())):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian channels for the last 20 periods (excluding current bar)
        highest_high = high[i-20:i].max()
        lowest_low = low[i-20:i].min()
        
        if position == 0:
            # LONG: Price breaks above upper Donchian band AND close > 1w EMA34 (bullish trend) AND volume confirm
            if (close[i] > highest_high and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian band AND close < 1w EMA34 (bearish trend) AND volume confirm
            elif (close[i] < lowest_low and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1w EMA34 (trend change)
            if close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 1w EMA34 (trend change)
            if close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals