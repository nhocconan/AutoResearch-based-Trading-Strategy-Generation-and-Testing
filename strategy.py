#!/usr/bin/env python3
"""
4h_1d_cci_extreme_v1
Hypothesis: CCI (Commodity Channel Index) on 1d timeframe identifies overbought/oversold extremes.
In strong trends, CCI > +100 or < -100 can persist, but reversals often occur at extremes.
We use: 
  - CCI(20) on 1d > +100 = potential short (mean reversion in overextended uptrend)
  - CCI(20) on 1d < -100 = potential long (mean reversion in overextended downtrend)
  - Entry only when 4h price closes back inside the extreme zone (i.e., CCI crosses back below +100 or above -100)
  - Volume confirmation: 4h volume > 1.5x 20-period average to avoid low-volume false signals
  - Trend filter: 4h EMA50 slope (rising/falling) to avoid counter-trend entries in strong momentum
Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
Works in bull/bear by fading extremes only when volume confirms and trend is not strongly opposing.
"""

name = "4h_1d_cci_extreme_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate CCI(20) on 1d
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    mad = np.where(mad == 0, 1e-10, mad)
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # Align CCI to 4h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    # 4h EMA50 for trend filter (slope)
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_slope = ema50 - np.roll(ema50, 1)
    ema50_slope[0] = 0
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after EMA50 warmup
        # Skip if CCI not ready
        if np.isnan(cci_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long setup: CCI was <-100 (oversold) and now crosses back above -100
        # Short setup: CCI was >+100 (overbought) and now crosses back below +100
        cci_now = cci_aligned[i]
        cci_prev = cci_aligned[i-1]
        
        # Long entry conditions
        long_setup = (cci_prev <= -100) and (cci_now > -100)
        long_entry = long_setup and vol_confirm[i] and (ema50_slope[i] > 0)  # only long if EMA50 rising
        
        # Short entry conditions
        short_setup = (cci_prev >= 100) and (cci_now < 100)
        short_entry = short_setup and vol_confirm[i] and (ema50_slope[i] < 0)  # only short if EMA50 falling
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or CCI crosses back to opposite extreme (optional early exit)
        elif position == 1 and (cci_now >= 100 or (cci_prev > -100 and cci_now <= -100)):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cci_now <= -100 or (cci_prev < 100 and cci_now >= 100)):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals