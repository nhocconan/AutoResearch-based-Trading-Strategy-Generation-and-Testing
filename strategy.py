#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w ADX(14) trend filter and volume confirmation
# Long when price breaks above 1d Donchian high(20) AND 1w ADX > 25 (trending) AND volume > 1.5 * avg_volume(20) on 12h
# Short when price breaks below 1d Donchian low(20) AND 1w ADX > 25 (trending) AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses back through the 1d Donchian midpoint (high+low)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian(20) provides strong breakout levels that reduce whipsaw
# 1w ADX(14) > 25 ensures we trade only in trending markets (works in both bull and bear)
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading

name = "12h_1dDonchian20_1wADX25_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed 1d bars for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Donchian High = rolling max of high(20)
    # Donchian Low = rolling min of low(20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high_1d = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_1d = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid_1d = (donchian_high_1d + donchian_low_1d) / 2.0
    
    # Align 1d Donchian to 12h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_1d)
    
    # Get 1w data ONCE before loop for ADX(14) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:  # Need at least 14 completed weekly bars for ADX(14)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
    # ADX calculation: +DM, -DM, TR, then smoothed averages, then DX, then ADX
    up_move = pd.Series(high_1w).diff()
    down_move = pd.Series(low_1w).diff()
    up_move = up_move.where(up_move > down_move, 0.0)
    down_move = (-down_move).where(down_move > up_move, 0.0)
    
    tr1 = pd.Series(high_1w) - pd.Series(low_1w)
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed values with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initial values (simple average of first period)
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    up_sum = up_move.rolling(window=period, min_periods=period).sum()
    down_sum = down_move.rolling(window=period, min_periods=period).sum()
    
    # Wilder's smoothing: today = (yesterday * (period-1) + today) / period
    atr = tr_sum.copy()
    plus_dm = up_sum.copy()
    minus_dm = down_sum.copy()
    
    for i in range(period, len(tr)):
        atr.iloc[i] = (atr.iloc[i-1] * (period-1) + tr.iloc[i]) / period
        plus_dm.iloc[i] = (plus_dm.iloc[i-1] * (period-1) + up_move.iloc[i]) / period
        minus_dm.iloc[i] = (minus_dm.iloc[i-1] * (period-1) + down_move.iloc[i]) / period
    
    # Avoid division by zero
    plus_di = 100 * plus_dm / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm / atr.replace(0, np.nan)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    
    # ADX is smoothed DX
    adx = dx.rolling(window=period, min_periods=period).mean()
    for i in range(period, len(dx)):
        if not np.isnan(dx.iloc[i]) and not np.isnan(adx.iloc[i-1]):
            adx.iloc[i] = (adx.iloc[i-1] * (period-1) + dx.iloc[i]) / period
    
    adx_1w = adx.values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian high, 1w ADX > 25 (trending), volume confirmation, in session
            if (close[i] > donchian_high_aligned[i] and 
                adx_1w_aligned[i] > 25.0 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low, 1w ADX > 25 (trending), volume confirmation, in session
            elif (close[i] < donchian_low_aligned[i] and 
                  adx_1w_aligned[i] > 25.0 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals