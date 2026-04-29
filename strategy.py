#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and volume confirmation
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when Bull Power < 0 AND Bear Power > 0 AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when Bull Power and Bear Power converge (both near zero) indicating weakening momentum
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Elder Ray measures bull/bear power relative to EMA13; confluence with 1d EMA50 filters weak signals.
# Volume spike ensures institutional participation, reducing false breakouts.
# Works in bull markets (strong bull power) and bear markets (strong bear power).

name = "6h_ElderRay_BullBearPower_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components (Bull Power and Bear Power) using 13-period EMA
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13) + 1  # EMA50 warmup + EMA13 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Bull Power and Bear Power converge (both near zero) indicating weakening momentum
            if abs(bull) < 0.1 and abs(bear) < 0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power and Bear Power converge (both near zero) indicating weakening momentum
            if abs(bull) < 0.1 and abs(bear) < 0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 AND volume confirmation
            if bull > 0 and bear < 0 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Bull Power < 0 AND Bear Power > 0 AND price < 1d EMA50 AND volume confirmation
            elif bull < 0 and bear > 0 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals