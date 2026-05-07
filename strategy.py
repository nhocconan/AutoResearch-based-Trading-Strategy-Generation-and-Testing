#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0, price > 1d EMA50, and volume > 1.5x 20-period EMA.
# Short when Bear Power < 0, Bull Power < 0, price < 1d EMA50, and volume > 1.5x 20-period EMA.
# Uses 1d EMA50 for trend direction and volume surge for momentum confirmation.
# Designed for moderate trade frequency (target: 20-40/year) to balance signal quality and cost.
# Works in bull markets via bull power strength and in bear markets via bear power dominance.
name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA50 for trend direction
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume surge: current volume > 1.5 * 20-period EMA
    vol_ema_20_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge_1d = np.where(vol_ema_20_1d > 0, df_1d['volume'].values / vol_ema_20_1d, 1.0) > 1.5
    vol_surge_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_surge_1d)
    
    # Calculate 13-period EMA for Elder Ray (standard period)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for EMA13 and 1d indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_surge_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: Bull Power > 0, Bear Power < 0, price > 1d EMA50, volume surge
            long_condition = (bull_power[i] > 0) and (bear_power[i] < 0) and (close[i] > ema50_1d_aligned[i]) and vol_surge_1d_aligned[i]
            # Short condition: Bear Power < 0, Bull Power < 0, price < 1d EMA50, volume surge
            short_condition = (bear_power[i] < 0) and (bull_power[i] < 0) and (close[i] < ema50_1d_aligned[i]) and vol_surge_1d_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bear Power becomes positive (bullish momentum fails) or price < 1d EMA50
            if (bear_power[i] >= 0) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bull Power becomes negative (bearish momentum fails) or price > 1d EMA50
            if (bull_power[i] <= 0) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals