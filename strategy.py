#!/usr/bin/env python3
name = "6h_Keltner_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(10) for Keltner channels
    tr = np.maximum(df_1d['high'].values - df_1d['low'].values,
                    np.maximum(np.abs(df_1d['high'].values - df_1d['close'].shift(1).values),
                               np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)))
    tr[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first TR
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate daily EMA(20) for Keltner center
    ema_20 = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    keltner_upper = ema_20 + 2.0 * atr
    keltner_lower = ema_20 - 2.0 * atr
    
    # Daily EMA(50) for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Keltner channels and daily EMA to 6h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Keltner upper with volume and in uptrend
            vol_condition = volume[i] > vol_ma[i] * 1.8
            uptrend = close[i] > ema_50_aligned[i]
            
            if close[i] > keltner_upper_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below Keltner lower with volume and in downtrend
            elif close[i] < keltner_lower_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below EMA(20) or volume drops
            if close[i] < ema_50_aligned[i] or volume[i] < vol_ma[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above EMA(20) or volume drops
            if close[i] > ema_50_aligned[i] or volume[i] < vol_ma[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Keltner breakout with daily trend filter and volume confirmation
# Keltner channels (EMA + ATR) adapt to volatility, providing dynamic support/resistance.
# Breaks above upper channel with volume in uptrend indicate strong momentum.
# Breaks below lower channel with volume in downtrend indicate strong selling pressure.
# Daily EMA(50) ensures trades align with intermediate-term trend.
# Works in bull (buy upper breaks in uptrend) and bear (sell lower breaks in downtrend).
# Position size 0.25 balances risk and keeps trade frequency ~15-35/year.