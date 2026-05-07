#!/usr/bin/env python3
name = "12h_Keltner_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h ATR(10) for Keltner channels
    tr1 = np.maximum(high, np.roll(close, 1))
    tr1 = np.maximum(tr1, np.roll(close, 1))
    tr2 = np.minimum(low, np.roll(close, 1))
    tr2 = np.minimum(tr2, np.roll(close, 1))
    tr = np.maximum(tr1 - tr2, np.abs(high - np.roll(low, 1)), np.abs(low - np.roll(high, 1)))
    tr[0] = high[0] - low[0]  # First bar
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 12h EMA(20) for Keltner center
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels
    keltner_upper = ema_20 + 2 * atr
    keltner_lower = ema_20 - 2 * atr
    
    # 12h volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA and ATR
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Keltner upper with volume and 1d uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]  # Rising EMA
            
            if close[i] > keltner_upper[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below Keltner lower with volume and 1d downtrend
            elif close[i] < keltner_lower[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below EMA(20) or volume drops
            if close[i] < ema_20[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above EMA(20) or volume drops
            if close[i] > ema_20[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Keltner channel breakout with 1d EMA trend filter and volume confirmation
# - Keltner channels (EMA + ATR*2) adapt to volatility better than fixed channels
# - 1d EMA(34) ensures alignment with higher timeframe trend
# - Volume spike (1.8x average) confirms institutional participation
# - Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Position size 0.25 targets 15-25 trades/year, avoiding fee drag
# - Exit at EMA(20) provides dynamic stop in trending markets