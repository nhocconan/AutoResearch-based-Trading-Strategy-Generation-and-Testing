#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and volume confirmation
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND close > 1d EMA50 AND volume > 1.5x 20-bar avg
# Short when Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND close < 1d EMA50 AND volume > 1.5x 20-bar avg
# Exit when momentum diverges (Bull Power < 0 for longs, Bear Power > 0 for shorts) OR price crosses 1d EMA50
# Uses discrete position sizing (0.25) to minimize fee churn.
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Works in bull markets (captures sustained bullish momentum) and bear markets (captures sustained bearish momentum).
# Volume filter ensures institutional participation, reducing false signals.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.

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
    
    # Calculate Elder Ray Bull/Bear Power using EMA(13) on 6h data
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
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
            # Exit: bull power turns negative OR price crosses below 1d EMA50
            if bull < 0 or curr_close < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bear power turns positive OR price crosses above 1d EMA50
            if bear > 0 or curr_close > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when bull power > 0 AND bear power < 0 (bullish momentum) AND close > 1d EMA50 AND volume confirmation
            if bull > 0 and bear < 0 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bear power < 0 AND bull power < 0 (bearish momentum) AND close < 1d EMA50 AND volume confirmation
            elif bear < 0 and bull < 0 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals