#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter.
# Long when: price > Donchian upper(20) AND volume > 1.5x volume MA(20) AND ADX(14) > 25
# Short when: price < Donchian lower(20) AND volume > 1.5x volume MA(20) AND ADX(14) > 25
# Exit when price crosses Donchian middle (20-period midpoint).
# Designed for 4h timeframe with moderate trade frequency (target: 25-40/year) to avoid fee drag.
# Uses volume confirmation to avoid false breakouts and ADX to ensure trending markets.
# Works in bull markets via upward breakouts, in bear markets via downward breakouts.
name = "4h_Donchian20_Volume_AdxTrend"
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
    
    # Donchian channels (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Volume confirmation: volume > 1.5x volume MA(20)
    volume_ma = np.full(n, np.nan)
    volume_series = pd.Series(volume)
    volume_ma_values = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ma = volume_ma_values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # ADX(14) for trend strength
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    plus_di = np.where(tr_sum > 0, 100 * plus_dm_sum / tr_sum, 0)
    minus_di = np.where(tr_sum > 0, 100 * minus_dm_sum / tr_sum, 0)
    
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_strong = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_ma[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high + volume confirmation + strong trend
            long_breakout = (close[i] > donchian_high[i]) and volume_confirm[i] and adx_strong[i]
            # Short breakdown: price < Donchian low + volume confirmation + strong trend
            short_breakout = (close[i] < donchian_low[i]) and volume_confirm[i] and adx_strong[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian middle
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian middle
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals