#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_Volume_Regime
1d strategy: Donchian(20) breakout with weekly trend filter, volume confirmation, and chop regime filter.
Long: Close breaks above 20-day high + weekly EMA20 > EMA50 + volume > 1.5x daily avg + chop < 61.8
Short: Close breaks below 20-day low + weekly EMA20 < EMA50 + volume > 1.5x daily avg + chop < 61.8
Exit: Opposite breakout or trend reversal
Designed for 10-20 trades/year per symbol (40-80 total over 4 years)
Works in trending markets via breakouts, avoids choppy regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 and EMA50 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) - chop > 61.8 = ranging, chop < 38.2 = trending
    def choppiness_index(high, low, close, window=14):
        atr = []
        tr = []
        for i in range(len(high)):
            if i == 0:
                tr.append(high[i] - low[i])
            else:
                tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
            atr_val = np.mean(tr[-window:]) if len(tr) >= window else np.nan
            atr.append(atr_val)
        atr = np.array(atr)
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        range_max_min = highest_high - lowest_low
        chop = 100 * np.log10(sum_atr / window) / np.log10(range_max_min)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for weekly EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20w_aligned[i]) or np.isnan(ema_50w_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend conditions
        weekly_uptrend = ema_20w_aligned[i] > ema_50w_aligned[i]
        weekly_downtrend = ema_20w_aligned[i] < ema_50w_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_roll[i]
        breakdown_down = close[i] < low_roll[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: only trade in trending markets (chop < 61.8)
        trending_regime = chop[i] < 61.8
        
        if position == 0:
            # Long: weekly uptrend + volume + breakout + trending regime
            if weekly_uptrend and vol_confirm and breakout_up and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + volume + breakdown + trending regime
            elif weekly_downtrend and vol_confirm and breakdown_down and trending_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal, volume breakdown, or chop regime shift
            if not weekly_uptrend or (vol_confirm and breakdown_down) or chop[i] >= 61.8:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal, volume breakout, or chop regime shift
            if not weekly_downtrend or (vol_confirm and breakout_up) or chop[i] >= 61.8:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_Regime"
timeframe = "1d"
leverage = 1.0