#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13) on 6h timeframe
# Long when Bull Power > 0 AND Bear Power rising (momentum) AND price > 1d EMA50 AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power falling (momentum) AND price < 1d EMA50 AND volume > 1.5x average
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.
# Primary timeframe: 6h, HTF: 1d for EMA50 trend filter.

name = "6h_ElderRay_Power_Trend_Volume"
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
    
    # Calculate 1d EMA50 for trend filter (loaded ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Elder Ray calculations on 6h timeframe
    # Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Momentum: rate of change of power (3-period)
    bull_power_momentum = bull_power - np.roll(bull_power, 3)
    bear_power_momentum = bear_power - np.roll(bear_power, 3)
    # Handle first 3 bars
    bull_power_momentum[:3] = 0
    bear_power_momentum[:3] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_momentum[i]) or np.isnan(bear_power_momentum[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND Bull Power momentum > 0 (rising) AND price > 1d EMA50 AND volume spike
            if (bull_power[i] > 0 and 
                bull_power_momentum[i] > 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bear Power momentum < 0 (falling) AND price < 1d EMA50 AND volume spike
            elif (bear_power[i] < 0 and 
                  bear_power_momentum[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 OR price < 1d EMA50
            if bull_power[i] <= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 OR price > 1d EMA50
            if bear_power[i] >= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals