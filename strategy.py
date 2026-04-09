#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w Donchian breakout with 1d ATR filter and volume confirmation
# 1w Donchian(20) provides major structural breakouts with proven edge in trending markets
# 1d ATR(14) filter ensures breakouts occur with sufficient volatility (>1.5x average)
# 4h volume confirmation (>2.0x 20-period average) filters low-quality breakouts
# Designed for 4h timeframe targeting 15-30 trades/year (60-120 over 4 years)
# Works in bull/bear: breakouts capture strong moves, volume/vol filters reduce false signals

name = "4h_1w_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channels (20-period)
    # Upper = highest high of last 20 weekly bars
    # Lower = lowest low of last 20 weekly bars
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Load 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr1_4h[0] = 0
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x average 4h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # 1d ATR filter: current 1d ATR > 1.5x average 1d ATR
        atr_confirmed = atr_1d_aligned[i] > 1.5 * np.nanmean(atr_1d_aligned[max(0, i-50):i+1])
        
        if position == 1:  # Long position
            # Update highest high since entry
            if close[i] > highest_since_long:
                highest_since_long = close[i]
            # ATR trailing stop: exit if price drops 3.0x ATR from highest
            if close[i] < highest_since_long - 3.0 * atr_4h[i]:
                position = 0
                highest_since_long = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if close[i] < lowest_since_short:
                lowest_since_short = close[i]
            # ATR trailing stop: exit if price rises 3.0x ATR from lowest
            if close[i] > lowest_since_short + 3.0 * atr_4h[i]:
                position = 0
                lowest_since_short = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Breakout at 1w Donchian levels with volume and ATR confirmation
            if volume_confirmed and atr_confirmed:
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    highest_since_long = close[i]
                    signals[i] = 0.30
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    lowest_since_short = close[i]
                    signals[i] = -0.30
    
    return signals