#!/usr/bin/env python3
"""
6h_Keltner_Breakout_Aroon_Volume_Filter
Hypothesis: Keltner Channel breakouts combined with Aroon trend strength and volume confirmation
capture momentum moves in both bull and bear markets. Aroon filters out weak breakouts,
ensuring only strong trends are traded. Volume confirms institutional participation.
Target: 15-25 trades/year (60-100 total over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Keltner and Aroon
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Keltner Channel (10, 1.5) - typical settings
    # Upper = EMA(10) + 1.5 * ATR(10)
    # Lower = EMA(10) - 1.5 * ATR(10)
    close_1d = df_1d['close']
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    
    # EMA(10)
    ema_10 = close_1d.ewm(span=10, adjust=False, min_periods=10).mean()
    
    # ATR(10)
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.ewm(span=10, adjust=False, min_periods=10).mean()
    
    keltner_up = ema_10 + 1.5 * atr_10
    keltner_low = ema_10 - 1.5 * atr_10
    
    # Aroon(25) - measures trend strength
    # Aroon Up = ((25 - days since 25-period high) / 25) * 100
    # Aroon Down = ((25 - days since 25-period low) / 25) * 100
    high_25 = high_1d.rolling(window=25, min_periods=25).max()
    low_25 = low_1d.rolling(window=25, min_periods=25).min()
    
    # Days since high/low
    since_high = 25 - high_1d.rolling(window=25, min_periods=1).apply(
        lambda x: 25 - np.argmax(x[::-1]) - 1, raw=True
    )
    since_low = 25 - low_1d.rolling(window=25, min_periods=1).apply(
        lambda x: 25 - np.argmin(x[::-1]) - 1, raw=True
    )
    
    aroon_up = ((25 - since_high) / 25) * 100
    aroon_down = ((25 - since_low) / 25) * 100
    
    # Align to 6h timeframe
    keltner_up_6h = align_htf_to_ltf(prices, df_1d, keltner_up.values)
    keltner_low_6h = align_htf_to_ltf(prices, df_1d, keltner_low.values)
    aroon_up_6h = align_htf_to_ltf(prices, df_1d, aroon_up.values)
    aroon_down_6h = align_htf_to_ltf(prices, df_1d, aroon_down.values)
    
    # Volume filter: >1.3x 20-period average (more lenient than 1.5x to increase trades slightly)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    start_idx = 25  # Warmup for Aroon
    
    for i in range(start_idx, n):
        if (np.isnan(keltner_up_6h[i]) or np.isnan(keltner_low_6h[i]) or
            np.isnan(aroon_up_6h[i]) or np.isnan(aroon_down_6h[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        kup = keltner_up_6h[i]
        klow = keltner_low_6h[i]
        aroon_up_val = aroon_up_6h[i]
        aroon_down_val = aroon_down_6h[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above upper Keltner with strong Aroon Up and volume
            if price > kup and aroon_up_val > 70 and aroon_down_val < 30 and vol_ok:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below lower Keltner with strong Aroon Down and volume
            elif price < klow and aroon_down_val > 70 and aroon_up_val < 30 and vol_ok:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 3 bars (1.5 days) to avoid whipsaw
            if bars_since_entry < 3:
                signals[i] = 0.25
            else:
                signals[i] = 0.25
                # Exit: price returns to middle of Keltner or Aroon weakens
                middle_keltner = (kup + klow) / 2
                if price < middle_keltner or aroon_up_val < 50:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 3 bars (1.5 days)
            if bars_since_entry < 3:
                signals[i] = -0.25
            else:
                signals[i] = -0.25
                # Exit: price returns to middle of Keltner or Aroon weakens
                middle_keltner = (kup + klow) / 2
                if price > middle_keltner or aroon_down_val < 50:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "6h_Keltner_Breakout_Aroon_Volume_Filter"
timeframe = "6h"
leverage = 1.0