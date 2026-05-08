#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R as momentum oscillator and 1w EMA as trend filter.
# Long when 1w EMA50 is up (bullish trend) and 1d Williams %R crosses above -50 from oversold.
# Short when 1w EMA50 is down (bearish trend) and 1d Williams %R crosses below -50 from overbought.
# Includes volume confirmation (volume > 1.5x 20-period average) to filter weak moves.
# Uses fixed position size of 0.25 to limit risk and reduce trade frequency.
# Target: 20-50 trades per year to avoid fee drag while capturing meaningful moves.

name = "4h_1wEMA50_1dWilliamsR_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1w EMA50 for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_up = ema_50_1w > np.roll(ema_50_1w, 1)  # Rising EMA
    ema_50_up[0] = False  # First value has no previous
    
    # 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # Williams %R signals: crossing above -50 (bullish) or below -50 (bearish)
    williams_above_50 = williams_r > -50
    williams_below_50 = williams_r < -50
    
    # Crossovers
    williams_cross_above = williams_above_50 & ~np.roll(williams_above_50, 1)
    williams_cross_above[0] = False
    williams_cross_below = williams_below_50 & ~np.roll(williams_below_50, 1)
    williams_cross_below[0] = False
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    # Align 1w EMA trend to 4h
    ema_50_up_aligned = align_htf_to_ltf(prices, df_1w, ema_50_up.astype(float))
    # Align 1d Williams %R crossovers to 4h
    williams_cross_above_aligned = align_htf_to_ltf(prices, df_1d, williams_cross_above.astype(float))
    williams_cross_below_aligned = align_htf_to_ltf(prices, df_1d, williams_cross_below.astype(float))
    # Align volume confirmation to 4h
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_up_aligned[i]) or np.isnan(williams_cross_above_aligned[i]) or
            np.isnan(williams_cross_below_aligned[i]) or np.isnan(vol_confirm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish trend (rising 1w EMA50) + Williams %R crosses above -50 + volume confirmation
            if (ema_50_up_aligned[i] and
                williams_cross_above_aligned[i] and
                vol_confirm_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish trend (falling 1w EMA50) + Williams %R crosses below -50 + volume confirmation
            elif ((not ema_50_up_aligned[i]) and
                  williams_cross_below_aligned[i] and
                  vol_confirm_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend turns bearish OR Williams %R crosses below -50 (overbought)
            if (not ema_50_up_aligned[i] or williams_cross_below_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend turns bullish OR Williams %R crosses above -50 (oversold)
            if (ema_50_up_aligned[i] or williams_cross_above_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals