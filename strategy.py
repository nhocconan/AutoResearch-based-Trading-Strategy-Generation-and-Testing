#!/usr/bin/env python3
# 4h_crsi_donchian_chop_v1
# Hypothesis: 4h strategy using Connors RSI (CRSI) for mean reversion signals,
# Donchian(20) breakouts for trend continuation, and Choppiness Index regime filter.
# In trending markets (CHOP < 38.2): trade Donchian breakouts with trend direction.
# In ranging markets (CHOP > 61.8): trade CRSI mean reversals at extremes.
# Volume confirmation filters false signals. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 20-40 trades/year. Works in both bull (trend following) and bear (mean reversion in ranges).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_crsi_donchian_chop_v1"
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
    
    # Daily HTF for Choppiness Index (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr1[0] = high_d[0] - low_d[0]  # first bar
    tr2[0] = high_d[0] - close_d[0]
    tr3[0] = low_d[0] - close_d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and Choppiness Index
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / (hh_14 - ll_14)) / np.log10(14)
    
    # Align Choppiness Index to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h indicators
    # Donchian(20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # RSI(3) for CRSI
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rsi_3 = 100 * (gain.rolling(window=3, min_periods=3).mean() / 
                    loss.rolling(window=3, min_periods=3).mean().replace(0, np.nan))
    rsi_3 = rsi_3.fillna(50).values  # neutral when no loss
    
    # RSI Streak(2) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    # RSI of streak
    streak_delta = pd.Series(streak).diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = -streak_delta.clip(upper=0)
    rsi_streak = 100 * (streak_gain.rolling(window=2, min_periods=2).mean() / 
                        streak_loss.rolling(window=2, min_periods=2).mean().replace(0, np.nan))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Percent Rank(100) - where close ranks in last 100 periods
    def percentile_of_score(arr, score):
        if len(arr) == 0 or np.isnan(score):
            return 50.0
        return (np.sum(arr < score) / len(arr)) * 100
    
    pct_rank = np.full(n, np.nan)
    for i in range(99, n):
        window = close[i-99:i+1]
        pct_rank[i] = percentile_of_score(window, close[i])
    
    # CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    crsi = (rsi_3 + rsi_streak + pct_rank) / 3.0
    
    # Volume confirmation (4h)
    volume_s = pd.Series(volume)
    volume_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(crsi[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(volume_ma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: CRSI > 50 (mean reversion) OR price below Donchian low
            if crsi[i] > 50 or close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CRSI < 50 (mean reversion) OR price above Donchian high
            if crsi[i] < 50 or close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Regime-based entries
                chop_val = chop_aligned[i]
                
                if chop_val < 38.2:  # Trending regime - follow Donchian breakouts
                    # Long: price breaks above Donchian high
                    if close[i] > donch_high[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price breaks below Donchian low
                    elif close[i] < donch_low[i]:
                        position = -1
                        signals[i] = -0.25
                        
                elif chop_val > 61.8:  # Ranging regime - mean reversion at CRSI extremes
                    # Long: CRSI oversold (< 15)
                    if crsi[i] < 15:
                        position = 1
                        signals[i] = 0.25
                    # Short: CRSI overbought (> 85)
                    elif crsi[i] > 85:
                        position = -1
                        signals[i] = -0.25
    
    return signals