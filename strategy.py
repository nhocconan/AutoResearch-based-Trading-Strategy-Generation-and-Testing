#!/usr/bin/env python3
"""
12h_WilliamsFractal_Breakout_1wTrend_VolumeChop_v1
Hypothesis: On 12h timeframe, Williams fractal breakouts (bullish/bearish) with 1-week EMA50 trend filter, volume confirmation (>2.0x 20-period average), and choppiness regime filter (CHOP < 50) produce high-quality trades. The weekly trend filter ensures alignment with the dominant multi-week momentum, while the chop filter avoids whipsaws in ranging markets. This strategy targets 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear regimes by only taking breakouts in the direction of the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Williams Fractals (requires 2-bar confirmation delay)
    high_1d = get_htf_data(prices, '1d')['high'].values
    low_1d = get_htf_data(prices, '1d')['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Align with 2 extra bars delay for fractal confirmation (needs 2 future 1d bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), bullish_fractal, additional_delay_bars=2)
    
    # 12h volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h choppiness index: CHOP > 61.8 = ranging (avoid breakouts), CHOP < 38.2 = trending (favor breakouts)
    # CHOP = 100 * log10(sum(ATR14) / (max(high,n) - min(low,n))) / log10(n)
    tr1 = np.maximum(high - low, np.absolute(high - np.concatenate([[np.nan], close[:-1]])))
    tr2 = np.maximum(tr1, np.absolute(low - np.concatenate([[np.nan], close[:-1]])))
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / (max_high_14 - min_low_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) == 0, 50, chop)  # avoid div by zero
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA, 20 for volume MA, 14*2 for chop, plus fractal delay)
    start_idx = max(50, 20, 28) + 2  # +2 for fractal confirmation delay
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation (stricter)
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Choppiness regime: only take breakouts when CHOP < 50 (less choppy/more trending)
        regime_ok = chop[i] < 50.0
        
        # Williams Fractal breakout conditions
        bullish_breakout = bullish_fractal_aligned[i] > 0  # actual fractal value (price level)
        bearish_breakout = bearish_fractal_aligned[i] > 0  # actual fractal value (price level)
        
        # Long logic: bullish fractal breakout in uptrend with volume and good regime
        if uptrend and volume_spike and bullish_breakout and regime_ok:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: bearish fractal breakout in downtrend with volume and good regime
        elif downtrend and volume_spike and bearish_breakout and regime_ok:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend OR regime becomes too choppy
        elif position == 1 and (not uptrend or chop[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend or chop[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1wTrend_VolumeChop_v1"
timeframe = "12h"
leverage = 1.0