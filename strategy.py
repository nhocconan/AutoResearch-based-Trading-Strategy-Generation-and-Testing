#!/usr/bin/env python3
"""
4h_TRIX_Zero_Cross_12hTrend_VolumeFilter
Hypothesis: Uses TRIX (12,9,9) zero crosses on 4h for momentum, filtered by 12h EMA50 trend and volume spikes.
TRIX is effective in both bull and bear markets as it filters noise and captures momentum shifts.
Volume spikes confirm breakout strength, reducing false signals.
Target: 20-40 trades per year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate TRIX on 4h: EMA(EMA(EMA(close, 12), 12), 12) then ROC
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # Volume spike (>1.6x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.6 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for TRIX and EMA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(trix[i]) or np.isnan(trix[i-1]):
            signals[i] = 0.0
            continue
        
        # Trend direction from 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # TRIX zero cross signals
        trix_cross_up = trix[i-1] <= 0 and trix[i] > 0
        trix_cross_down = trix[i-1] >= 0 and trix[i] < 0
        
        # Entry logic:
        # Long: TRIX crosses above zero in uptrend with volume
        long_entry = vol_confirm and trend_up and trix_cross_up
        # Short: TRIX crosses below zero in downtrend with volume
        short_entry = vol_confirm and trend_down and trix_cross_down
        
        # Exit logic: Opposite TRIX cross or trend reversal
        long_exit = trix_cross_down or not trend_up
        short_exit = trix_cross_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_TRIX_Zero_Cross_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0