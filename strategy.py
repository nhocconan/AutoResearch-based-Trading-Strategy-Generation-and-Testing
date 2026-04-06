#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily volume confirmation and ADX trend filter
# Enter long on breakout above 20-period high when ADX > 25 and volume > 1.5x average
# Enter short on breakdown below 20-period low when ADX > 25 and volume > 1.5x average
# Exit when price returns to opposite Donchian band or ADX < 20 (trend weakening)
# Uses daily trend filter to avoid counter-trend trades in choppy markets
# Target: 60-120 trades over 4 years (15-30/year) with focus on strong trending moves

name = "12h_donchian20_1d_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # ADX for trend strength (14-period)
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First element has no previous close
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
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Daily volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_values[i]) or np.isnan(volume_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian band OR ADX < 20 (trend weakening)
            if close[i] <= donchian_low[i] or adx_values[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian band OR ADX < 20 (trend weakening)
            if close[i] >= donchian_high[i] or adx_values[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + ADX > 25 + volume confirmation
            volume_ok = volume[i] > 1.5 * volume_ma_1d_aligned[i]
            adx_strong = adx_values[i] > 25
            
            if volume_ok and adx_strong:
                if close[i] > donchian_high[i]:
                    # Breakout above upper band - bullish
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i]:
                    # Breakdown below lower band - bearish
                    signals[i] = -0.25
                    position = -1
    
    return signals