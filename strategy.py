#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ADX regime and Elder Ray calculation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
- Regime: ADX > 25 = trending (use Elder Ray), ADX <= 25 = ranging (fade extremes).
- Entry: Long when Bull Power > 0 and rising + volume spike in trending market.
         Short when Bear Power < 0 and falling + volume spike in trending market.
         In ranging market: Long at Bear Power extreme (oversold), Short at Bull Power extreme (overbought).
- Exit: Opposite Elder Ray signal or mean reversion to EMA13.
- Works in bull via buying strength, in bear via selling weakness, and in range via mean reversion.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_elder_ray(high, low, close, ema_len=13):
    """Calculate Elder Ray Bull and Bear Power"""
    ema_close = pd.Series(close).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean()
    bull_power = high - ema_close.values
    bear_power = low - ema_close.values
    return bull_power, bear_power, ema_close.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(0, high[i] - high[i-1])
        minus_dm[i] = max(0, low[i-1] - low[i])
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        elif plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        else:
            minus_dm[i] = 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    atr = np.zeros_like(high)
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    
    atr[period] = np.mean(tr[1:period+1])
    plus_dm_sm = np.sum(plus_dm[1:period+1])
    minus_dm_sm = np.sum(minus_dm[1:period+1])
    
    for i in range(period+1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_sm = (plus_dm_sm * (period-1) + plus_dm[i]) / period
        minus_dm_sm = (minus_dm_sm * (period-1) + minus_dm[i]) / period
        plus_di[i] = 100 * plus_dm_sm / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sm / atr[i] if atr[i] != 0 else 0
    
    dx = np.zeros_like(high)
    adx = np.zeros_like(high)
    for i in range(period*2, len(high)):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    adx[period*2] = np.mean(dx[period*2:period*2+period])
    for i in range(period*2+1, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ADX and EMA
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    
    # Calculate 1d Elder Ray (Bull/Bear Power) and EMA13
    bull_power_1d, bear_power_1d, ema13_1d = calculate_elder_ray(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align 1d indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1d bars for ADX/EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(ema13_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        ema13 = ema13_aligned[i]
        
        if position == 0:
            # Trending market (ADX > 25): follow Elder Ray momentum
            if adx_val > 25:
                # Long when Bull Power > 0 and rising (current > previous)
                if bull_power > 0 and i > start_idx and bull_power > bull_power_aligned[i-1]:
                    if volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                # Short when Bear Power < 0 and falling (current < previous)
                elif bear_power < 0 and i > start_idx and bear_power < bear_power_aligned[i-1]:
                    if volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
            # Ranging market (ADX <= 25): mean reversion at Elder Ray extremes
            else:
                # Long when Bear Power is extremely negative (oversold)
                if bear_power < -0.5 * np.std(bear_power_aligned[max(0, i-50):i]):  # 0.5 std below mean
                    if volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                # Short when Bull Power is extremely positive (overbought)
                elif bull_power > 0.5 * np.std(bull_power_aligned[max(0, i-50):i]):  # 0.5 std above mean
                    if volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Bear Power turns negative or mean reversion to EMA13
            if bear_power < 0 or close[i] >= ema13:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power turns positive or mean reversion to EMA13
            if bull_power > 0 or close[i] <= ema13:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADXRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0