#!/usr/bin/env python3
name = "6h_KeltnerBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50) and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    
    # ATR(14) for Keltner channels
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Keltner channels on 1d: EMA(20) ± 2*ATR(14)
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper_1d = ema20_1d + 2 * atr14
    kc_lower_1d = ema20_1d - 2 * atr14
    
    # Align 1d indicators to 6h
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    kc_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, kc_upper_1d)
    kc_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, kc_lower_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and ATR
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trend_up_1d_aligned[i]) or np.isnan(kc_upper_1d_aligned[i]) or
            np.isnan(kc_lower_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Keltner upper + 1d uptrend + volume confirmation
            if close[i] > kc_upper_1d_aligned[i] and trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Keltner lower + 1d downtrend + volume confirmation
            elif close[i] < kc_lower_1d_aligned[i] and not trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Keltner lower OR 1d trend turns down
            if close[i] < kc_lower_1d_aligned[i] or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Keltner upper OR 1d trend turns up
            if close[i] > kc_upper_1d_aligned[i] or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals