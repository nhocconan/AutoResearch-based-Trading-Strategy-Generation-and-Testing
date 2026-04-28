#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA(close), Bear Power = Low - EMA(close).
# Enter long when Bull Power > 0 and rising (current > previous) with volume > 1.5x 20-bar average and close > 1d EMA34.
# Enter short when Bear Power < 0 and falling (current < previous) with volume > 1.5x average and close < 1d EMA34.
# Exit when Elder Power crosses zero (Bull Power <= 0 for long exit, Bear Power >= 0 for short exit).
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Works in bull markets (strong Bull Power with uptrend) and bear markets (strong Bear Power with downtrend).
# Uses 1d EMA34 for trend filter (reduces whipsaws) and Elder Ray for momentum measurement.

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray components on 6h timeframe
    # Bull Power = High - EMA(close)
    # Bear Power = Low - EMA(close)
    close_series = pd.Series(close)
    ema_close = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values  # 13-bar EMA for Elder Ray
    
    bull_power = high - ema_close
    bear_power = low - ema_close
    
    # Calculate volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d EMA34 bias
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        # Elder Ray conditions: momentum and direction
        # Bull Power rising: current > previous
        bull_power_rising = i > 0 and bull_power[i] > bull_power[i-1]
        # Bear Power falling: current < previous
        bear_power_falling = i > 0 and bear_power[i] < bear_power[i-1]
        
        # Entry conditions
        long_entry = (bull_power[i] > 0) and bull_power_rising and vol_confirm and bullish_bias
        short_entry = (bear_power[i] < 0) and bear_power_falling and vol_confirm and bearish_bias
        
        # Exit conditions: Elder Power crosses zero
        long_exit = bull_power[i] <= 0
        short_exit = bear_power[i] >= 0
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals