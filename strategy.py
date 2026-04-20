#!/usr/bin/env python3
# 12h_1w_KAMA_Direction_With_Filter
# Hypothesis: 12h KAMA direction filtered by 1w trend and volume confirmation on 12h.
# Uses KAMA(12h) for trend direction, filters with 1w EMA trend, and requires volume spike.
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull/bear via 1w trend filter - only trade with the weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_KAMA_Direction_With_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h and 1w data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_12h) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # === 12h: KAMA for trend direction ===
    close_12h = df_12h['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.abs(np.diff(close_12h))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # === 1w: EMA for trend filter ===
    close_1w = df_1w['close'].values
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume_12h = df_12h['volume'].values
    vol_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = np.divide(volume_12h, vol_ma20_12h, out=np.full_like(volume_12h, np.nan), where=vol_ma20_12h!=0)
    
    # Align all to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        ema_30_1w_val = ema_30_1w_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(ema_30_1w_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA, weekly uptrend, volume confirmation
            if (close_val > kama_val and 
                ema_30_1w_val > close_1w[-1] if len(close_1w) > 0 else False and  # Weekly EMA trending up
                vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, weekly downtrend, volume confirmation
            elif (close_val < kama_val and 
                  ema_30_1w_val < close_1w[-1] if len(close_1w) > 0 else False and  # Weekly EMA trending down
                  vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below KAMA or weekly trend changes
            if close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above KAMA or weekly trend changes
            if close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals