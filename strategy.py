#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA Trend Filter and Volume Confirmation
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) measures bull/bear strength relative to trend
- 12h EMA(50) filter ensures alignment with intermediate trend for multi-timeframe confirmation
- Volume > 1.5x 20-period average confirms momentum with moderate filtering
- Designed for 6h timeframe targeting 12-25 trades/year (50-100 over 4 years) to minimize fee drag
- Works in bull markets via strong Bull Power + uptrend, in bear markets via strong Bear Power + downtrend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe (completed 12h bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA(13) on 6h close for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA(13)
    bear_power = low - ema_13   # Low - EMA(13)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # 12h EMA needs ~50 bars, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray signals with trend filter and volume confirmation
        # Long: Bull Power > 0 (bulls in control) + price above 12h EMA50 (uptrend) + volume confirmation
        # Short: Bear Power < 0 (bears in control) + price below 12h EMA50 (downtrend) + volume confirmation
        long_signal = (bull_power[i] > 0 and 
                      close[i] > ema_50_aligned[i] and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (bear_power[i] < 0 and 
                       close[i] < ema_50_aligned[i] and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: power weakening or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power <= 0 (bulls losing control) or price below 12h EMA50 (trend break)
                if (bull_power[i] <= 0 or 
                    close[i] < ema_50_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Bear Power >= 0 (bears losing control) or price above 12h EMA50 (trend break)
                if (bear_power[i] >= 0 or 
                    close[i] > ema_50_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0