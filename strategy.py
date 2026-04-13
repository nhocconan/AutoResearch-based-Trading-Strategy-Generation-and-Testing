#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with daily volume confirmation and ADX trend filter.
# Uses daily Donchian channels for structure, daily volume for conviction, and daily ADX to avoid ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.
# Designed to work in both bull (breakouts) and bear (avoid false breakouts in ranges via ADX).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channel on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ADX for trend strength
    # ADX calculation: +DM, -DM, TR, then smoothed
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff()
    tr = pd.Series(np.maximum(np.abs(tr1), np.abs(tr2))).fillna(0)
    
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_sum = tr.rolling(window=14, min_periods=14).sum()
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    plus_di = 100 * (plus_dm_sum / tr_sum)
    minus_di = 100 * (minus_dm_sum / tr_sum)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 4-hour timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 1.5x daily volume MA (adjusted for 4h)
        # 6 4h periods per day, so daily MA/6 = approximate 4h period MA
        volume_4h_approx_ma = volume_ma_20_1d_aligned[i] / 6
        volume_condition = volume[i] > (volume_4h_approx_ma * 1.5)
        
        # ADX condition: avoid ranging markets (ADX < 25 indicates weak trend)
        adx_condition = adx_aligned[i] > 25
        
        # Entry conditions: Donchian breakout with volume and ADX filter
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        
        if position == 0:
            if breakout_long and volume_condition and adx_condition:
                position = 1
                signals[i] = position_size
            elif breakout_short and volume_condition and adx_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price breaks below Donchian low or ADX falls below 20 (trend weakening)
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price breaks above Donchian high or ADX falls below 20 (trend weakening)
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0