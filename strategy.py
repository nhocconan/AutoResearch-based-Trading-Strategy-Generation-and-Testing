#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using weekly Bollinger Band breakout with 1d volume spike and ADX trend filter
# Long when price breaks above weekly BB upper band AND ADX(14) > 25 AND volume > 1.5 * avg_volume(20) on 4h
# Short when price breaks below weekly BB lower band AND ADX(14) > 25 AND volume > 1.5 * avg_volume(20) on 4h
# Exit when price crosses back inside weekly Bollinger Bands OR ADX drops below 20
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-150 total trades over 4 years (19-37/year) for 4h timeframe
# Weekly Bollinger Bands provide dynamic support/resistance from higher timeframe
# ADX filter ensures we only trade in trending markets, reducing whipsaw
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "4h_WeeklyBB_Breakout_ADX25_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least one completed weekly bar for BB
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    close_1w_series = pd.Series(close_1w)
    bb_middle = close_1w_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_1w_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    
    # Align weekly Bollinger Bands to 4h timeframe (wait for completed weekly bar)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    
    # Get 4h data ONCE before loop for ADX and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ADX(14) on 4h data
    adx_period = 14
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    # Smoothed values
    atr = pd.Series(tr).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Align 4h ADX to 4h timeframe (no additional delay needed)
    adx_aligned = adx  # Already on 4h timeframe
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_4h > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly BB upper, ADX > 25, volume confirmation, in session
            if close[i] > bb_upper_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly BB lower, ADX > 25, volume confirmation, in session
            elif close[i] < bb_lower_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses back inside weekly BB OR ADX drops below 20
            if close[i] < bb_upper_aligned[i] and close[i] > bb_lower_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses back inside weekly BB OR ADX drops below 20
            if close[i] < bb_upper_aligned[i] and close[i] > bb_lower_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals