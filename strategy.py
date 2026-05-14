#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Uses Donchian channel breakouts from prior 1d for structure, volume > 1.5x 20-bar average for conviction,
# and 1w EMA34 > EMA89 for bullish trend (or reverse for bearish) to ensure alignment with weekly momentum.
# Discrete position sizing (0.0, ±0.25) minimizes fee churn. Designed to capture strong breakouts
# in trending markets while avoiding false signals in ranging conditions. Targets 15-25 trades/year per symbol.

name = "1d_Donchian20_Breakout_1wEMATrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # --- 1d Indicators ---
    # Donchian channel (20) from prior bar
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: > 1.5x 20-bar average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # EMA34 and EMA89 on weekly
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1w = pd.Series(close_1w).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align to 1d (wait for completed 1w bar)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema89_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema89_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: bullish if EMA34 > EMA89, bearish if EMA34 < EMA89
        is_bullish_trend = ema34_1w_aligned[i] > ema89_1w_aligned[i]
        is_bearish_trend = ema34_1w_aligned[i] < ema89_1w_aligned[i]
        
        if position == 0:
            # Look for new entries only in alignment with weekly trend
            if is_bullish_trend and close[i] > highest_20[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            elif is_bearish_trend and close[i] < lowest_20[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: weekly trend turns bearish OR price retouches midpoint
            weekly_mid = (highest_20[i] + lowest_20[i]) / 2.0
            if not is_bullish_trend or close[i] < weekly_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly trend turns bullish OR price retouches midpoint
            weekly_mid = (highest_20[i] + lowest_20[i]) / 2.0
            if not is_bearish_trend or close[i] > weekly_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals