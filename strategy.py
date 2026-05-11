#!/usr/bin/env python3
"""
6h_Keltner_Channel_MeanReversion
Hypothesis: In the 6h timeframe, price often reverts to the mean after touching the Keltner Channel bands (EMA20 ± 2*ATR(10)). 
We add a 1d trend filter (price above/below 1d EMA50) to only take mean-reversion trades in the direction of the higher timeframe trend, 
which improves win rate in both bull and bear markets. Volume confirmation (current volume > 1.5x 20-period average) filters low-quality signals.
Targets 15-25 trades/year via strict entry conditions combining Keltner touch, 1d trend, and volume.
"""

name = "6h_Keltner_Channel_MeanReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 6h EMA20 and ATR(10) for Keltner Channel ---
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean()
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.ewm(span=10, adjust=False, min_periods=10).mean()
    
    upper_keltner = (ema_20 + 2 * atr_10).values
    lower_keltner = (ema_20 - 2 * atr_10).values
    ema_20_arr = ema_20.values
    
    # --- 1d EMA50 Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema_20_arr[i]) or np.isnan(ema_50_6h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price touches lower Keltner band + above 1d EMA50 + volume
            if (low[i] <= lower_keltner[i] and 
                close[i] > ema_50_6h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price touches upper Keltner band + below 1d EMA50 + volume
            elif (high[i] >= upper_keltner[i] and 
                  close[i] < ema_50_6h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price returns to EMA20 (mean reversion complete)
            if position == 1:
                # Exit long: price crosses above EMA20
                if close[i] > ema_20_arr[i] and close[i-1] <= ema_20_arr[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses below EMA20
                if close[i] < ema_20_arr[i] and close[i-1] >= ema_20_arr[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals