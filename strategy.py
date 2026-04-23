#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + ADX regime filter + volume confirmation.
- Bull Power = High - EMA13 (1d), Bear Power = EMA13 (1d) - Low
- Long when Bull Power > 0 AND ADX(1d) > 25 (trending) AND volume > 1.5x 20-period average
- Short when Bear Power > 0 AND ADX(1d) > 25 AND volume > 1.5x 20-period average
- Exit when power becomes negative (Bull Power <= 0 for long, Bear Power <= 0 for short) OR ADX < 20 (range)
- Uses 1d HTF for Elder Ray and ADX to avoid lower TF noise; 6h for execution timing.
- Target: 80-180 total trades over 4 years (20-45/year) for 6h timeframe.
- Works in both bull (strong Bull/Bear Power) and bear (ADX filters whipsaws, power persists in trends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d
    bear_power = ema_13_1d - low_1d
    
    # Calculate 1d ADX (trend strength)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align all 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d volume average for spike filter
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 14)  # volume MA, EMA13, ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bp = bull_power_aligned[i]
        br = bear_power_aligned[i]
        adx_val = adx_1d_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        vol_current = df_1d['volume'].values[min(i, len(df_1d['volume'].values)-1)] if hasattr(df_1d['volume'], 'values') else volume[i]  # fallback
        
        # Use 6h volume for more timely confirmation
        vol_current = volume[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND ADX > 25 (strong trend) AND volume spike
            if bp > 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND ADX > 25 AND volume spike
            elif br > 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 OR ADX < 20 (losing trend)
                if bp <= 0 or adx_val < 20:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power <= 0 OR ADX < 20
                if br <= 0 or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_ADX_Regime_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0