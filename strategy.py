#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ADX(14) trend filter
# Long when price breaks above Donchian upper band + volume spike + ADX>25
# Short when price breaks below Donchian lower band + volume spike + ADX>25
# Exit when price crosses midline or ADX<20 (trend weakening)
# Works in bull (breakouts with momentum) and bear (breakdowns with momentum)
# Uses discrete sizing (0.25) to limit overtrading and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initial values
    atr[13] = np.mean(tr[1:14])
    plus_dm_smooth[13] = np.mean(plus_dm[1:14])
    minus_dm_smooth[13] = np.mean(minus_dm[1:14])
    
    # Wilder's smoothing
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # DI and DX
    plus_di = np.where(atr != 0, plus_dm_smooth / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_dm_smooth / atr * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    
    # ADX
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX value
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: break above upper band + volume + ADX>25
        if (close[i] > donch_high[i] and 
            volume[i] > vol_threshold[i] and 
            adx_aligned[i] > 25):
            signals[i] = 0.25
        
        # Short: break below lower band + volume + ADX>25
        elif (close[i] < donch_low[i] and 
              volume[i] > vol_threshold[i] and 
              adx_aligned[i] > 25):
            signals[i] = -0.25
        
        # Exit: price crosses midline OR ADX<20 (trend weakening)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] <= donch_mid[i] or adx_aligned[i] < 20)) or
               (signals[i-1] == -0.25 and (close[i] >= donch_mid[i] or adx_aligned[i] < 20)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0