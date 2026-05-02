#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d EMA34 trend filter and volume spike confirmation.
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and increasing, Bear Power < 0 and decreasing, price > 1d EMA34, volume spike.
# Short when Bear Power < 0 and decreasing, Bull Power > 0 and increasing, price < 1d EMA34, volume spike.
# Uses 1d HTF for EMA34 trend filter to reduce whipsaw in ranging markets.
# Discrete sizing 0.25 to minimize fee churn. Target: 50-150 trades over 4 years.
# Primary timeframe: 6h, HTF: 1d for EMA34.

name = "6h_ElderRay_Power_1dEMA34_Volume_Trend"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray Power: EMA13 of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA13 and EMA34)
    start_idx = 35
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 and rising, Bear Power < 0 and falling, price > 1d EMA34, volume spike
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and
                bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and
                close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, Bull Power > 0 and rising, price < 1d EMA34, volume spike
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and
                  bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and
                  close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 or Bear Power >= 0 or price < 1d EMA34
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 or Bull Power <= 0 or price > 1d EMA34
            if (bear_power[i] >= 0 or bull_power[i] <= 0 or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals