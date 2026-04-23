#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
Long when Tenkan crosses above Kijun AND price > Cloud AND close > 1d EMA34 AND volume > 1.5x 20-period average.
Short when Tenkan crosses below Kijun AND price < Cloud AND close < 1d EMA34 AND volume > 1.5x 20-period average.
Exit when Tenkan/Kijun cross reverses OR price retraces to Kijun line.
Uses discrete position sizing (0.25) targeting ~12-30 trades/year on 6h timeframe.
Ichimoku provides dynamic support/resistance via Cloud and momentum via TK cross.
Works in bull (trend-following with Cloud support) and bear (mean-reversion at Cloud resistance via TK cross).
"""

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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Cloud top/bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 34, 20)  # Ichimoku needs 52, EMA34 needs 34, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        ema34_val = ema34_1d_aligned[i]
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price > Cloud AND uptrend (price > EMA34) AND volume spike
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and  # TK cross up
                price > cloud_top[i] and close[i] > ema34_val and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price < Cloud AND downtrend (price < EMA34) AND volume spike
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and  # TK cross down
                  price < cloud_bottom[i] and close[i] < ema34_val and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Tenkan/Kijun cross reverses
            if position == 1 and tenkan[i] < kijun[i]:
                exit_signal = True
            elif position == -1 and tenkan[i] > kijun[i]:
                exit_signal = True
            
            # Secondary exit: Price retraces to Kijun line
            if position == 1 and close[i] <= kijun[i]:
                exit_signal = True
            elif position == -1 and close[i] >= kijun[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_TK_Cross_Cloud_Filter_1dEMA34_Trend_VolumeConfirmation_TKExit_KijunExit"
timeframe = "6h"
leverage = 1.0