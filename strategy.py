#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA13) flipped? Actually:
# Bull Power = High - EMA13, Bear Power = Low - EMA13
# We use: Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 AND volume > 1.5x avg
# Short when Bull Power < 0 AND Bear Power > 0 AND price < 1d EMA50 AND volume > 1.5x avg
# Elder Ray measures bull/bear strength relative to EMA; combined with 1d trend filter avoids counter-trend trades.
# Volume confirmation ensures breakout legitimacy. Discrete sizing (0.25) controls fee drag.
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
    
    # Calculate EMA(13) for Elder Ray on 6h data
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20)  # EMA13 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_50 = ema_50_1d_aligned[i]
        curr_close = close[i]
        bp = bull_power[i]
        br = bear_power[i]
        
        # Handle exits: reverse when Elder Ray signals change
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 (weakening bullish strength)
            if bp <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 (weakening bearish strength)
            if br >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 AND volume confirmation
            if bp > 0 and br < 0 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Bull Power < 0 AND Bear Power > 0 AND price < 1d EMA50 AND volume confirmation
            elif bp < 0 and br > 0 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals