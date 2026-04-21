#!/usr/bin/env python3
"""
4h_4HR_Momentum_Breakout_1dTrend_Volume
Hypothesis: Use 4-hour momentum breakout above 20-period high + 1d EMA50 trend filter + volume confirmation. Designed to capture momentum bursts in trending markets with volume surge confirming institutional interest. Works in bull/bear markets by following higher timeframe trend (1d EMA) while using 4h momentum for precise entry. Target 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h momentum: 20-period high breakout ===
    high_4h = prices['high'].values
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    
    # === Volume confirmation: 20-period volume average on 4h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_50_1d_aligned[i]
        high_breakout = high_20[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above 20-period high + volume spike > 1.5 + price above 1d EMA50
            if (price_close > high_breakout and 
                vol_spike > 1.5 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low + volume spike > 1.5 + price below 1d EMA50
            else:
                low_20 = pd.Series(prices['low'].values).rolling(window=20, min_periods=20).min().values[i]
                if (price_close < low_20 and 
                    vol_spike > 1.5 and 
                    price_close < trend_1d):
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when price returns to 10-period moving average
            ma_10 = pd.Series(prices['close'].values).rolling(window=10, min_periods=10).mean().values[i]
            if position == 1 and price_close < ma_10:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ma_10:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_4HR_Momentum_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0