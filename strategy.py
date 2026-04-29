#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 AND increasing AND price > 1d EMA50 AND volume > 2x 20-bar avg
# Short when Bear Power > 0 AND increasing AND price < 1d EMA50 AND volume > 2x 20-bar avg
# Exit when respective power becomes negative (momentum loss)
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Elder Ray measures bull/bear strength relative to EMA13; rising power indicates strengthening trend.
# 1d EMA50 filters counter-trend moves, volume confirmation ensures participation.
# Works in bull markets (rising Bull Power) and bear markets (rising Bear Power).

name = "6h_ElderRay_BullBearPower_1dEMA50_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components: EMA13 of close
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 50)  # EMA13 warmup and EMA50 alignment
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_50 = ema_50_1d_aligned[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        prev_bull = bull_power[i-1]
        prev_bear = bear_power[i-1]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Bull Power turns negative (momentum loss)
            if curr_bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns negative (momentum loss)
            if curr_bear <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND increasing AND price > 1d EMA50 AND volume confirmation
            if curr_bull > 0 and curr_bull > prev_bull and close[i] > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power > 0 AND increasing AND price < 1d EMA50 AND volume confirmation
            elif curr_bear > 0 and curr_bear > prev_bear and close[i] < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals