#!/usr/bin/env python3
"""
12h_ThreeBarBreakout_1dTrend_Volume
Hypothesis: Detect early trend continuation on 12h timeframe using 3-bar breakout pattern (close > 3-bar high for long, close < 3-bar low for short) with 1d EMA34 trend filter and volume confirmation. This captures momentum bursts while avoiding false breakouts in chop. Works in bull markets via trend-following breakouts and in bear markets via short-side breakdowns. Target 12-37 trades/year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume confirmation: 20-period volume average on 12h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    # === 3-bar breakout levels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 3-bar high/low (excluding current bar)
    high_3bar = np.maximum.reduce([np.roll(high, 1), np.roll(high, 2), np.roll(high, 3)])
    low_3bar = np.minimum.reduce([np.roll(low, 1), np.roll(low, 2), np.roll(low, 3)])
    # Handle NaN from rolling
    high_3bar[:3] = np.nan
    low_3bar[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(high_3bar[i]) or
            np.isnan(low_3bar[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Close breaks above 3-bar high + volume spike > 1.5 + price above 1d EMA34
            if (price_close > high_3bar[i] and 
                vol_spike > 1.5 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below 3-bar low + volume spike > 1.5 + price below 1d EMA34
            elif (price_close < low_3bar[i] and 
                  vol_spike > 1.5 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to opposite 3-bar level (mean reversion within the breakout range)
            if position == 1 and price_close < low_3bar[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > high_3bar[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_ThreeBarBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0