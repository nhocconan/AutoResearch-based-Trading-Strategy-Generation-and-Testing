#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_KAMA_Trend_PriceAction_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA trend (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on weekly close
    # Efficiency ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)  # will fix below
    # Proper ER calculation
    er = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        if i == 0:
            er[i] = 0
        else:
            direction = np.abs(close_1w[i] - close_1w[i-10]) if i >= 10 else np.abs(close_1w[i] - close_1w[0])
            volatility = np.sum(np.abs(np.diff(close_1w[max(0,i-9):i+1])) if i >= 1 else np.abs(close_1w[i] - close_1w[i-1]))
            er[i] = direction / volatility if volatility > 0 else 0
    
    # Smoothing constants
    sc = (er * 0.6 + 0.064) ** 2  # 2 to 30 EMA equivalent
    
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to 12h
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # 12h price action: higher highs/lows for trend confirmation
    # Higher high: current high > previous high
    hh = high > np.roll(high, 1)
    # Higher low: current low > previous low
    hl = low > np.roll(low, 1)
    # Lower high: current high < previous high
    lh = high < np.roll(high, 1)
    # Lower low: current low < previous low
    ll = low < np.roll(low, 1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough for weekly alignment and volume
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama_val = kama_1w_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trend: price relative to weekly KAMA
        above_kama = price > kama_val
        below_kama = price < kama_val
        
        # Price action confirmation
        strong_uptrend = hh[i] and hl[i]  # Higher high and higher low
        strong_downtrend = lh[i] and ll[i]  # Lower high and lower low
        
        if position == 0:
            # Long: above KAMA + uptrend price action + volume
            if above_kama and strong_uptrend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: below KAMA + downtrend price action + volume
            elif below_kama and strong_downtrend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: below KAMA or breakdown in price action
            if below_kama or (lh[i] and ll[i]):  # Lower high and lower low = breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: above KAMA or reversal in price action
            if above_kama or (hh[i] and hl[i]):  # Higher high and higher low = reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals