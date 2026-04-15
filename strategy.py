#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smoothing(dx, period)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 6h Donchian channel (20-period)
    # Upper = highest high over 20 periods, Lower = lowest low over 20 periods
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_upper = rolling_max(high, 20)
    donchian_lower = rolling_min(low, 20)
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_6h[i]) or np.isnan(ema_34_6h[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: ADX > 25 (trending) + price > EMA34 (uptrend) + break above Donchian upper + volume confirmation
        # Short: ADX > 25 (trending) + price < EMA34 (downtrend) + break below Donchian lower + volume confirmation
        # Volume: > 1.5x average
        # Discrete position sizing: 0.25
        
        # Long conditions
        if (adx_6h[i] > 25 and                    # Strong trend
            close[i] > ema_34_6h[i] and             # Price above EMA34 (uptrend)
            close[i] > donchian_upper[i] and        # Break above Donchian upper (20-period high)
            volume_ratio[i] > 1.5):                 # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions
        elif (adx_6h[i] > 25 and                  # Strong trend
              close[i] < ema_34_6h[i] and           # Price below EMA34 (downtrend)
              close[i] < donchian_lower[i] and      # Break below Donchian lower (20-period low)
              volume_ratio[i] > 1.5):               # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ADX_EMA34_Donchian_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0