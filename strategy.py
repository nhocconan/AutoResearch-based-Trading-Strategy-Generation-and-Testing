#!/usr/bin/env python3
name = "6h_ADX_Alligator_ElderRay_Trend_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d data for ADX, Alligator, Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX (14) on 1d
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            plus_di[i] = 100 * (plus_dm[i] / atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * (minus_dm[i] / atr[i]) if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Alligator (Jaws, Teeth, Lips) on 1d
    def alligator_lines(close, jaw_period=13, teeth_period=8, lips_period=5):
        jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(8).values
        teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(5).values
        lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().shift(3).values
        return jaw, teeth, lips
    
    jaw_1d, teeth_1d, lips_1d = alligator_lines(close_1d)
    
    # Elder Ray (Bull Power, Bear Power) on 1d
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align all 1d indicators to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 25, Alligator bullish (Lips > Teeth > Jaws), Bull Power > 0, volume spike
            if (adx_1d_aligned[i] > 25 and 
                lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i] and 
                bull_power_1d_aligned[i] > 0 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25, Alligator bearish (Lips < Teeth < Jaws), Bear Power < 0, volume spike
            elif (adx_1d_aligned[i] > 25 and 
                  lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i] and 
                  bear_power_1d_aligned[i] < 0 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR ADX drops below 20
            if (lips_1d_aligned[i] < teeth_1d_aligned[i] or adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR ADX drops below 20
            if (lips_1d_aligned[i] > teeth_1d_aligned[i] or adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals