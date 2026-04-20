#!/usr/bin/env python3
# 12h_1w_KAMA_Trend_Follow
# Hypothesis: 12h KAMA trend with 1w trend filter and volume confirmation. Trades only with higher timeframe trend to avoid whipsaws.
# Uses KAMA crossover on 12h for entry, 1w EMA for trend filter, and volume spike for confirmation. Designed for low trade frequency.
# Works in bull/bear via 1w trend filter - only trade when 1w trend aligns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_KAMA_Trend_Follow"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1w: EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 12h: KAMA (ER=10) for trend and entry ===
    close_12h = prices['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)  # 10-period sum of absolute changes
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # start at index 9
    for i in range(10, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Get values
        close_val = close_12h[i]
        kama_val = kama[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(ema_34_1w_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA above price (bullish) AND price above 1w EMA (uptrend filter) AND volume confirmation
            if (kama_val > close_val and  # KAMA above price = bullish
                close_val > ema_34_1w_val and  # Price above 1w EMA = uptrend
                vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: KAMA below price (bearish) AND price below 1w EMA (downtrend filter) AND volume confirmation
            elif (kama_val < close_val and  # KAMA below price = bearish
                  close_val < ema_34_1w_val and  # Price below 1w EMA = downtrend
                  vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA crosses below price (trend change) OR volume dies
            if kama_val < close_val:  # KAMA crossed below price
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA crosses above price (trend change) OR volume dies
            if kama_val > close_val:  # KAMA crossed above price
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals