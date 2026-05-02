#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike and ADX trend filter
# Donchian channel breakouts capture strong momentum moves. Volume confirmation ensures
# breakouts are supported by participation. ADX > 25 filters for trending markets
# to avoid false breakouts in ranging conditions. Designed for 75-200 total trades
# over 4 years (19-50/year) on 4h timeframe. Works in bull markets (buying breakouts
# in uptrend) and bear markets (selling breakdowns in downtrend) by taking trades
# only when ADX confirms trending regime.

name = "4h_Donchian20_VolumeSpike_ADX25_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ADX for trend filter (14-period)
    # ADX requires +DI, -DI, and TR calculation
    # +DM = high[i] - high[i-1] if high[i] - high[i-1] > low[i-1] - low[i] else 0
    # -DM = low[i-1] - low[i] if low[i-1] - low[i] > high[i] - high[i-1] else 0
    # TR = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    # +DI = 100 * smoothed +DM / ATR
    # -DI = 100 * smoothed -DM / ATR
    # ADX = smoothed DX where DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    
    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Smoothed +DM and -DM
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    for i in range(1, n):
        plus_dm_smooth[i] = (1 - alpha) * plus_dm_smooth[i-1] + alpha * plus_dm[i]
        minus_dm_smooth[i] = (1 - alpha) * minus_dm_smooth[i-1] + alpha * minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di = np.where(atr > 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr > 0, 100 * minus_dm_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx = np.zeros(n)
    adx[0] = dx[0]
    for i in range(1, n):
        adx[i] = (1 - alpha) * adx[i-1] + alpha * dx[i]
    
    # Donchian channel (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for ADX and Donchian)
    start_idx = max(34, 20)  # 34 for ADX, 20 for Donchian
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high with volume spike AND ADX > 25 (trending)
            if (close[i] > donchian_high[i] and 
                volume_spike[i] and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume spike AND ADX > 25 (trending)
            elif (close[i] < donchian_low[i] and 
                  volume_spike[i] and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (failed breakout) OR ADX < 20 (trend weakening)
            if close[i] < donchian_low[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (failed breakdown) OR ADX < 20 (trend weakening)
            if close[i] > donchian_high[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals