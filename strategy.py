#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with daily trend filter and volume confirmation.
Bull Power = Close - EMA13, Bear Power = EMA13 - Low.
In bull market (daily close > daily EMA50): long when Bull Power > 0 and rising.
In bear market (daily close < daily EMA50): short when Bear Power > 0 and rising.
Volume must be above 20-period average to confirm strength.
Elder Ray measures bull/bear power behind moves, effective in both trends.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)  # already shifted
    
    # === ELDER RAY (LTF) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema13  # Close - EMA13
    bear_power = ema13 - low    # EMA13 - Low
    
    # Smooth power for rising detection (3-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Rising detection: current > previous
    bull_rising = bull_power_smooth > np.roll(bull_power_smooth, 1)
    bear_rising = bear_power_smooth > np.roll(bear_power_smooth, 1)
    bull_rising[0] = False
    bear_rising[0] = False
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(daily_ema_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA
        bull_trend = close[i] > daily_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 OR trend turns bearish
            if bull_power[i] <= 0 or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power <= 0 OR trend turns bullish
            if bear_power[i] <= 0 or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on daily trend
            if bull_trend:
                # In bull market: long when Bull Power > 0 and rising
                if bull_power[i] > 0 and bull_rising[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short when Bear Power > 0 and rising
                if bear_power[i] > 0 and bear_rising[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals