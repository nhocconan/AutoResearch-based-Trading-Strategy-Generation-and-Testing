#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_With_Volume_Filtered"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 1d data for Camarilla pivots (from previous 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d bar's range
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla R1 and S1 levels
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (using previous 1d bar's values)
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: current volume > 2.0x 20-period average (more stringent)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Choppiness regime filter: avoid choppy markets
    # Calculate 14-period chop: higher = choppier
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # True range over 14 periods for denominator
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.where((max_high - min_low) > 0, max_high - min_low, 1)
    chop = 100 * np.log10(atr * 14 / chop_denom) / np.log10(14)
    chop = np.where(chop_denom > 0, chop, 50)  # default to neutral
    
    # Only trade in trending markets (CHOP < 38.2) or strong mean reversion (CHOP > 61.8)
    # For breakout strategy, we prefer trending markets
    trend_regime = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(trend_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above 1d EMA50 (uptrend) AND volume surge AND trending regime
            if close[i] > r1_4h[i] and close[i] > ema_1d_aligned[i] and volume_filter[i] and trend_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND below 1d EMA50 (downtrend) AND volume surge AND trending regime
            elif close[i] < s1_4h[i] and close[i] < ema_1d_aligned[i] and volume_filter[i] and trend_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 OR below 1d EMA50 (trend change) OR volume drops
            if close[i] < s1_4h[i] or close[i] < ema_1d_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R1 OR above 1d EMA50 (trend change) OR volume drops
            if close[i] > r1_4h[i] or close[i] > ema_1d_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals