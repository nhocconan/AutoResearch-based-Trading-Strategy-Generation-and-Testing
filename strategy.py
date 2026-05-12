#!/usr/bin/env python3
# 1h_VolumeSpike_4hTrend_1dTrend
# Hypothesis: Enter on volume spikes (vol > 2x MA) only when 4h and 1d trends align, using 1h for timing.
# Long: Price > 4h EMA20 AND > 1d EMA50 AND volume spike. Short: Price < both EMAs AND volume spike.
# Exit when volume drops below spike threshold or trend misaligns.
# Uses higher timeframes for direction (1-3 trades/week expected) and 1h for precise entry.
# Volume spike filter reduces false signals; trend alignment works in bull/bear markets.

name = "1h_VolumeSpike_4hTrend_1dTrend"
timeframe = "1h"
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
    
    # === 4h data for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # === 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 4h EMA20 for short-term trend
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d EMA50 for long-term trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detector: volume > 2x 24-period MA
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend alignment: both 4h and 1d agree
        bullish_alignment = (close[i] > ema_20_4h_aligned[i]) and (close[i] > ema_50_1d_aligned[i])
        bearish_alignment = (close[i] < ema_20_4h_aligned[i]) and (close[i] < ema_50_1d_aligned[i])
        
        if position == 0:
            # Enter long on volume spike with bullish alignment
            if bullish_alignment and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Enter short on volume spike with bearish alignment
            elif bearish_alignment and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long if volume drops below spike or trend misaligns
            if not vol_spike[i] or not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short if volume drops below spike or trend misaligns
            if not vol_spike[i] or not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals