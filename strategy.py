#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d EMA(50) trend filter and volume spike confirmation.
# Elder Ray Power (Bull Power = High - EMA, Bear Power = Low - EMA) identifies buying/selling pressure.
# Bull Power > 0 + rising indicates bullish momentum; Bear Power < 0 + falling indicates bearish momentum.
# Combined with 1d EMA trend filter to avoid counter-trend trades and volume spike for confirmation.
# Designed for 6h timeframe to target 15-35 trades/year per symbol.
# Works in bull/bear via trend filter + momentum-based entry.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h EMA(22) for Elder Ray Power
    ema_22 = pd.Series(close).ewm(span=22, adjust=False, min_periods=22).mean().values
    
    # Elder Ray Power components
    bull_power = high - ema_22  # Buying pressure
    bear_power = low - ema_22   # Selling pressure
    
    # Slope of Elder Ray Power (momentum) - 3-period slope
    bull_power_slope = pd.Series(bull_power).diff(3).values
    bear_power_slope = pd.Series(bear_power).diff(3).values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(bull_power_slope[i]) or
            np.isnan(bear_power_slope[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising + uptrend (close > daily EMA50) + volume spike
            if (bull_power[i] > 0 and bull_power_slope[i] > 0 and 
                close[i] > ema_50_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling + downtrend (close < daily EMA50) + volume spike
            elif (bear_power[i] < 0 and bear_power_slope[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on Bull Power <= 0 or trend reversal
                if (bull_power[i] <= 0 or close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Bear Power >= 0 or trend reversal
                if (bear_power[i] >= 0 or close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0