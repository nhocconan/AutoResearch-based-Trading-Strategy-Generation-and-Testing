#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Chop_12hTrend_v1
Hypothesis: 4h Donchian(20) breakout with volume confirmation, choppiness regime filter, and 12h EMA50 trend filter.
- Long when price breaks above Donchian(20) high AND volume spike AND chop < 61.8 (trending) AND 12h EMA50 uptrend
- Short when price breaks below Donchian(20) low AND volume spike AND chop < 61.8 (trending) AND 12h EMA50 downtrend
- Exit on opposite Donchian(10) break or trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Donchian channels (20 for entry, 10 for exit)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume spike (2.0x 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    # Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr3[0] = 0  # First bar has no previous close
    atr14 = np.zeros(n)
    for i in range(n):
        tr_vals = [tr1[i], tr2[i], tr3[i]]
        if i < 14:
            atr14[i] = np.mean(tr_vals[:i+1]) if i >= 0 else 0
        else:
            atr14[i] = np.mean(tr_vals)
    # Smoothed ATR (Wilder's smoothing)
    atr14_smooth = np.zeros(n)
    atr14_smooth[13] = np.mean(atr14[:14])
    for i in range(14, n):
        atr14_smooth[i] = (atr14_smooth[i-1] * 13 + atr14[i]) / 14
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros(n)
    for i in range(n):
        if i >= 13 and atr14_smooth[i] > 0 and (highest_high_14[i] - lowest_low_14[i]) > 0:
            sum_atr = atr14_smooth[i] * 14
            chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(highest_high_14[i] - lowest_low_14[i])
        else:
            chop[i] = 50  # Neutral value when not enough data
    
    # Trending regime: CHOP < 61.8
    trending_regime = chop < 61.8
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA, 14 for chop)
    start_idx = max(20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(volume_spike[i]) or np.isnan(trending_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Entry conditions
        if position == 0:
            # Long: Donchian breakout up AND volume spike AND trending regime AND 12h EMA50 uptrend
            if close[i] > donchian_high_20[i] and volume_spike[i] and trending_regime[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND volume spike AND trending regime AND 12h EMA50 downtrend
            elif close[i] < donchian_low_20[i] and volume_spike[i] and trending_regime[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Donchian(10) breakdown OR trend reversal (price < EMA50)
            if close[i] < donchian_low_10[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Donchian(10) breakout OR trend reversal (price > EMA50)
            if close[i] > donchian_high_10[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Chop_12hTrend_v1"
timeframe = "4h"
leverage = 1.0