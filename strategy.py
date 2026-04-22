#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with weekly ADX trend filter and volume confirmation
# Uses Donchian channel breakouts for trend following, weekly ADX to filter strong trends only,
# volume to confirm breakout strength, and ATR-based stoploss to manage risk.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for Donchian calculation (to avoid look-ahead)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period Donchian channels on 12h data
    high_max_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ADX for trend strength filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range and Directional Movement for ADX
    tr1 = np.abs(np.diff(high_1w, prepend=high_1w[0]))
    tr2 = np.abs(np.diff(low_1w, prepend=low_1w[0]))
    tr3 = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(tr)
    minus_di = np.zeros_like(tr)
    
    atr[0] = tr[0]
    plus_di[0] = plus_dm[0]
    minus_di[0] = minus_dm[0]
    
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
        plus_di[i] = (1 - alpha) * plus_di[i-1] + alpha * plus_dm[i]
        minus_di[i] = (1 - alpha) * minus_di[i-1] + alpha * minus_dm[i]
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = np.zeros_like(dx)
    adx[atr_period-1] = np.mean(dx[:atr_period]) if len(dx) >= atr_period else 0
    
    for i in range(atr_period, len(dx)):
        adx[i] = (1 - alpha) * adx[i-1] + alpha * dx[i]
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to main timeframe
    high_max_20_aligned = align_htf_to_ltf(prices, df_12h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_12h, low_min_20)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + volume spike + strong uptrend (ADX > 25)
            if (close[i] > high_max_20_aligned[i] and vol_spike[i] and adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + volume spike + strong downtrend (ADX > 25)
            elif (close[i] < low_min_20_aligned[i] and vol_spike[i] and adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or ADX weakens
            if position == 1:
                if (close[i] < low_min_20_aligned[i] or adx_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > high_max_20_aligned[i] or adx_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_WeeklyADX_Trend_Volume_Session"
timeframe = "12h"
leverage = 1.0