#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h EMA50 Trend + Volume Spike
# Long when Bull Power > 0 AND price > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short when Bear Power < 0 AND price < 12h EMA50 AND volume > 2.0x 20-bar avg
# Exit when power reverses sign (Bull Power <= 0 for long, Bear Power >= 0 for short)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 6h timeframe.
# Elder Ray measures bull/bear power via EMA13, 12h EMA50 filters trend direction,
# volume confirmation ensures breakout strength. Works in bull/bear via trend filter.

name = "6h_ElderRay_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h data
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray on 6h data using EMA13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # volume MA and EMA13 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50 = ema_50_12h_aligned[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 (bulls losing control)
            if curr_bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 (bears losing control)
            if curr_bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND price > 12h EMA50 AND volume confirmation
            if curr_bull > 0 and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power < 0 AND price < 12h EMA50 AND volume confirmation
            elif curr_bear < 0 and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals