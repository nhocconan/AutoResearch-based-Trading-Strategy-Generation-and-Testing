#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX25 regime filter and volume confirmation.
# Long when Bull Power > 0 AND ADX > 25 (trending market) AND volume spike (>1.5x 20-period volume MA).
# Short when Bear Power < 0 AND ADX > 25 AND volume spike.
# Uses 1d EMA13 for trend direction (price > EMA13 = uptrend for long, price < EMA13 = downtrend for short).
# Elder Ray measures trend strength via price-EMA relationship. ADX filters for trending regimes only.
# Volume spike confirms institutional participation. Designed for 6h timeframe to achieve 50-150 total trades over 4 years.
# Works in both bull and bear markets by only trading in the direction of the 1d trend when ADX confirms strength.

name = "6h_ElderRay_1dADX25_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Get 1d data for Elder Ray calculation and ADX filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13_1d
    bear_power = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray components to lower timeframe (1d -> 6h)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Align EMA13 for trend direction
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(ema_13_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        open_val = open_prices[i]
        vol_spike = volume_spike[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        adx_val = adx_aligned[i]
        ema13_val = ema_13_1d_aligned[i]
        
        # Trend direction from EMA13
        trend_up = close_val > ema13_val   # 1d uptrend
        trend_down = close_val < ema13_val  # 1d downtrend
        
        if position == 0:
            # Long: Bull Power > 0 AND ADX > 25 AND uptrend AND volume spike
            if bull_val > 0 and adx_val > 25 and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND ADX > 25 AND downtrend AND volume spike
            elif bear_val < 0 and adx_val > 25 and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: Bull Power <= 0 (loss of bullish momentum)
            if bull_val <= 0:
                exit_signal = True
            # Exit: ADX <= 25 (trend weakening)
            elif adx_val <= 25:
                exit_signal = True
            # Exit: Trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: Bear Power >= 0 (loss of bearish momentum)
            if bear_val >= 0:
                exit_signal = True
            # Exit: ADX <= 25 (trend weakening)
            elif adx_val <= 25:
                exit_signal = True
            # Exit: Trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals