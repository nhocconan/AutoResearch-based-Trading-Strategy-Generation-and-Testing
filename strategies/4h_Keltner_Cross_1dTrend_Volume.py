#!/usr/bin/env python3
name = "4h_Keltner_Cross_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and Keltner basis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA(20) for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d ATR(10) for Keltner channels
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner channels: middle = EMA(20), width = 2 * ATR(10)
    keltner_middle = ema_1d
    keltner_upper = keltner_middle + 2 * atr_1d
    keltner_lower = keltner_middle - 2 * atr_1d
    
    # Align Keltner channels to 4h
    keltner_middle_aligned = align_htf_to_ltf(prices, df_1d, keltner_middle)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    
    # Volume filter: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA(20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(keltner_middle_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above upper Keltner with uptrend and volume
            if (close[i] > keltner_upper_aligned[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close below lower Keltner with downtrend and volume
            elif (close[i] < keltner_lower_aligned[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below middle Keltner or trend change
            if close[i] < keltner_middle_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above middle Keltner or trend change
            if close[i] > keltner_middle_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Keltner channel breakout with 1d EMA(20) trend filter and volume confirmation.
# Keltner channels adapt to volatility, providing dynamic support/resistance.
# In uptrends, buy breakouts above upper channel; in downtrends, sell breakdowns below lower channel.
# Exit when price returns to middle channel or trend changes.
# Volume filter ensures trades occur with participation. Position size 0.25 controls risk.
# Works in both bull (trend following) and bear (counter-trend reversals at extremes) markets.