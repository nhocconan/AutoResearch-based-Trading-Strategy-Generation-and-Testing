#!/usr/bin/env python3
"""
12h_WilliamsFractal_1dTrend_VolumeConfirm
Hypothesis: Uses daily Williams Fractals for reversal signals, filtered by 1-week trend and volume confirmation. Works in both bull and bear markets by capturing mean-reversion at key swing points. Designed for 12h timeframe to maintain low trade frequency (<50/year) while capturing significant reversals.
"""

name = "12h_WilliamsFractal_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals (need 5 bars: 2 left, center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    
    # Williams fractals need 2 extra bars for confirmation (center + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Get 1w trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    cooldown = 0  # cooldown to prevent whipsaw
    
    for i in range(50, n):
        # Decrease cooldown if active
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            # LONG: Bullish fractal (support) with volume confirmation in uptrend
            if (bullish_fractal_aligned[i] > 0 and not np.isnan(bullish_fractal_aligned[i]) and
                low[i] <= bullish_fractal_aligned[i] and volume_confirmed[i] and
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish fractal (resistance) with volume confirmation in downtrend
            elif (bearish_fractal_aligned[i] > 0 and not np.isnan(bearish_fractal_aligned[i]) and
                  high[i] >= bearish_fractal_aligned[i] and volume_confirmed[i] and
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches bearish fractal (resistance) or trend weakens
            if (bearish_fractal_aligned[i] > 0 and not np.isnan(bearish_fractal_aligned[i]) and
                high[i] >= bearish_fractal_aligned[i]) or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 2  # 2-bar cooldown after exit
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches bullish fractal (support) or trend weakens
            if (bullish_fractal_aligned[i] > 0 and not np.isnan(bullish_fractal_aligned[i]) and
                low[i] <= bullish_fractal_aligned[i]) or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 2  # 2-bar cooldown after exit
            else:
                signals[i] = -0.25
    
    return signals