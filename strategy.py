#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses weekly EMA34 to establish medium-term trend direction (bull/bear).
# Breakouts above 20-day high or below 20-day low with volume (>1.5x 20-bar avg) capture momentum.
# 1w EMA34 filter ensures alignment with weekly trend to reduce whipsaws in ranging markets.
# Volume confirmation filters low-participation breakouts.
# Discrete sizing (0.25) minimizes fee churn. Target: 50-100 total trades over 4 years (12-25/year).
# Works in both bull (breakouts with trend) and bear (mean reversion at extremes via opposite breakouts).

name = "1d_Donchian20_WeeklyEMA34_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) on 1w close
    ema_1w_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Donchian(20) channels: 20-period high/low
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_34_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_conf = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 20-day high, above 1w EMA, and volume confirmation
            if curr_high > high_ma_20[i] and curr_close > ema_1w_34_aligned[i] and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low, below 1w EMA, and volume confirmation
            elif curr_low < low_ma_20[i] and curr_close < ema_1w_34_aligned[i] and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below 20-day low or below 1w EMA
            if curr_low < low_ma_20[i] or curr_close < ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above 20-day high or above 1w EMA
            if curr_high > high_ma_20[i] or curr_close > ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals