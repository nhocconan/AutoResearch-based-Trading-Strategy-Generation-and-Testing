#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend direction.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h).
- Trend filter: Only trade long when price > 12h EMA50, short when price < 12h EMA50.
- Volume confirmation: Current volume > 1.5 * 20-period average volume.
- Entry: Long when Bull Power > 0 AND price > 12h EMA50 AND volume confirmation.
         Short when Bear Power < 0 AND price < 12h EMA50 AND volume confirmation.
- Exit: Opposite signal (Bear Power >= 0 for long exit, Bull Power <= 0 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets via long signals, in bear markets via short signals, avoids whipsaws via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 20-period average volume for confirmation
    if len(prices) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 50, 20)  # Need 13 for EMA13, 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price > 12h EMA50 for long bias, price < 12h EMA50 for short bias
        long_bias = curr_close > ema50_12h_aligned[i]
        short_bias = curr_close < ema50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Exit conditions: opposite Elder Ray signal
        if position != 0:
            # Exit long: Bear Power >= 0 (bullish momentum fading)
            if position == 1:
                if bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bull Power <= 0 (bearish momentum fading)
            elif position == -1:
                if bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray with trend and volume filters
        if position == 0:
            # Long: Bull Power > 0 AND long bias AND volume confirmation
            long_condition = (bull_power[i] > 0 and 
                            long_bias and
                            volume_confirm)
            
            # Short: Bear Power < 0 AND short bias AND volume confirmation
            short_condition = (bear_power[i] < 0 and 
                             short_bias and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0