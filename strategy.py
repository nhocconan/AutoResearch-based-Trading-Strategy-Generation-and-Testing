#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA34_Trend_VolumeConfirm_ChopFilter
Hypothesis: Trade daily Donchian(20) breakouts with 1-week EMA34 trend filter, volume confirmation (>1.5x 20-bar MA), and choppiness regime filter (CHOP < 61.8 for trending markets). 
This strategy targets trending markets on the daily timeframe to capture medium-term moves while avoiding false breakouts in ranging conditions. 
Discrete sizing 0.25 balances profit and fee drag. Target: 15-25 trades/year (~60-100 over 4 years) to stay within fee drag limits for 1d timeframe.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period) from daily data
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Choppiness regime filter: CHOP < 61.8 indicates trending market (use 14-period)
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of absolute price changes over 14 periods
    abs_changes = np.abs(np.diff(close, prepend=close[0]))
    sum_abs_changes = pd.Series(abs_changes).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_abs_changes / (atr_14 * 14)) / np.log10(10)
    chop_regime = chop < 61.8  # Trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1w EMA34 (34), Donchian (20), volume MA (20), and CHOP (14)
    start_idx = max(34, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian Upper AND 1w trend bullish (close > EMA34) AND volume confirm AND trending regime
            long_setup = (close[i] > donchian_upper[i]) and \
                         (close[i] > ema_34_1w_aligned[i]) and \
                         volume_confirm[i] and \
                         chop_regime[i]
            # Short: price breaks below Donchian Lower AND 1w trend bearish (close < EMA34) AND volume confirm AND trending regime
            short_setup = (close[i] < donchian_lower[i]) and \
                          (close[i] < ema_34_1w_aligned[i]) and \
                          volume_confirm[i] and \
                          chop_regime[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel OR 1w trend turns bearish OR chop regime turns ranging
            if (close[i] < donchian_upper[i] and close[i] > donchian_lower[i]) or \
               (close[i] < ema_34_1w_aligned[i]) or \
               (not chop_regime[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR 1w trend turns bullish OR chop regime turns ranging
            if (close[i] < donchian_upper[i] and close[i] > donchian_lower[i]) or \
               (close[i] > ema_34_1w_aligned[i]) or \
               (not chop_regime[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeConfirm_ChopFilter"
timeframe = "1d"
leverage = 1.0