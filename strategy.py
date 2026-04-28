#!/usr/bin/env python3
"""
4h_RSI_Extreme_4hTrend_Filter_VolumeSpike
Hypothesis: Use 4h RSI extremes (overbought/oversold) with 4h EMA200 trend filter and volume spike confirmation.
Works in both bull and bear markets by fading extremes only when aligned with trend.
Targets 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA200 for trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 4h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume spike confirmation (2.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_200[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Trend direction from EMA200
        trend_up = close[i] > ema_200[i]
        trend_down = close[i] < ema_200[i]
        
        # Volume confirmation: >2.5x 20-period MA
        vol_confirm = volume[i] > (2.5 * vol_ma_20[i])
        
        # RSI extremes
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        
        # Entry conditions: fade extremes only with trend alignment
        long_entry = vol_confirm and trend_up and rsi_oversold
        short_entry = vol_confirm and trend_down and rsi_overbought
        
        # Exit conditions: RSI returns to neutral zone or trend reversal
        long_exit = (rsi_values[i] > 50) or (not trend_up)
        short_exit = (rsi_values[i] < 50) or (not trend_down)
        
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

name = "4h_RSI_Extreme_4hTrend_Filter_VolumeSpike"
timeframe = "4h"
leverage = 1.0