#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Regime_v1
Hypothesis: 12h Camarilla R1/S1 breakout with 1w trend filter and choppiness regime. Only trade breakouts in direction of 1w trend when market is not too choppy (CHOP < 61.8). Uses volume confirmation to avoid false breakouts. Designed for BTC/ETH - Camarilla pivots work in both bull/bear markets via trend alignment and regime filter. Target: 50-150 total trades over 4 years (12-37/year) by requiring 1w trend alignment, regime filter, and volume spike.
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
    
    # Load 1w and 1d data ONCE before loop for HTF trend and regime
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1w EMA34 for HTF trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    htf_trend = np.where(close > ema_34_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate choppiness index on 1d
    def choppiness_index(high, low, close, window=14):
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.close.shift(1))), np.abs(low - np.close.shift(1)))).rolling(window=window, min_periods=window).sum()
        max_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        min_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(window)
        return chop.values
    
    # Fix: use previous close for true range calculation
    close_shift = np.roll(close, 1)
    close_shift[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - close_shift)
    tr3 = np.abs(low - close_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop[np.isnan(chop)] = 100  # default to choppy when not enough data
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels from previous 1d
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_range = df_1d['high'] - df_1d['low']
    r1 = df_1d['close'] + 1.1 * camarilla_range / 12
    s1 = df_1d['close'] - 1.1 * camarilla_range / 12
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 14 for CHOP, 20 for volume MA)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade when not too choppy (CHOP < 61.8 = trending)
        not_choppy = chop_aligned[i] < 61.8
        
        # Volume spike condition
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_long = close[i] > r1_aligned[i]
        breakout_short = close[i] < s1_aligned[i]
        
        if htf_trend[i] == 1 and not_choppy:  # Uptrend and trending regime
            # Long signal: breakout above R1 with volume spike
            if breakout_long and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long: breakout below S1
            elif breakout_short:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1 and not_choppy:  # Downtrend and trending regime
            # Short signal: breakout below S1 with volume spike
            if breakout_short and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short: breakout above R1
            elif breakout_long:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # In choppy regime or counter-trend: stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Regime_v1"
timeframe = "12h"
leverage = 1.0