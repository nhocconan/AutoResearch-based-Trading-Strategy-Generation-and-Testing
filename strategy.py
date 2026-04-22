#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA trend filter + volume confirmation
# Williams Alligator (Jaws/Teeth/Lips) identifies trend: Lips above Teeth above Jaws = uptrend, reverse = downtrend.
# 1d EMA50 acts as higher-timeframe trend filter to avoid counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms momentum and reduces false signals.
# Designed for 6h timeframe targeting 20-40 trades/year, works in bull/bear via trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA filter and volume average (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume 20-period average for spike detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Williams Alligator on 6h data (13,8,5 smoothed with 3,5,8 periods)
    # Jaws: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods  
    # Lips: 5-period SMMA smoothed by 3 periods
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(smma(close, 13), 8)
    teeth = smma(smma(close, 8), 5)
    lips = smma(smma(close, 5), 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaws (bullish alignment) + price > 1d EMA50 + volume spike
            if (lips[i] > teeth[i] > jaws[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaws (bearish alignment) + price < 1d EMA50 + volume spike
            elif (lips[i] < teeth[i] < jaws[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines cross or trend reversal
            if position == 1:
                # Exit on Alligator death cross (Lips < Jaws) or price below 1d EMA50
                if (lips[i] < jaws[i] or 
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Alligator death cross (Lips > Jaws) or price above 1d EMA50
                if (lips[i] > jaws[i] or 
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0