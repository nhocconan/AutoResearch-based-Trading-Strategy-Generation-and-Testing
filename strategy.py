#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with weekly trend filter and volume confirmation
# Uses 20-period Donchian channels on 12h timeframe with weekly ADX trend filter
# Only trades in direction of weekly trend to avoid counter-trend whipsaws
# Volume confirmation ensures breakout strength
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# Works in bull/bear via trend filter - only trades with weekly trend

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ADX for trend strength
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Load 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels (20-period)
    high_ma20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_ma20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align weekly indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di)
    
    # Align 12h indicators to 12h timeframe (no additional delay needed)
    high_ma20_aligned = align_htf_to_ltf(prices, df_12h, high_ma20)
    low_ma20_aligned = align_htf_to_ltf(prices, df_12h, low_ma20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i]) or np.isnan(high_ma20_aligned[i]) or
            np.isnan(low_ma20_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + weekly uptrend (ADX>25 and +DI>-DI) + volume spike
            if (close[i] > high_ma20_aligned[i] and adx_aligned[i] > 25 and 
                plus_di_aligned[i] > minus_di_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + weekly downtrend (ADX>25 and -DI>+DI) + volume spike
            elif (close[i] < low_ma20_aligned[i] and adx_aligned[i] > 25 and 
                  minus_di_aligned[i] > plus_di_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < low_ma20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_ma20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_WeeklyADX_Trend_Volume_Session"
timeframe = "12h"
leverage = 1.0