#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d ADX Regime with Volume Spike Confirmation
- Primary timeframe: 6h for execution, HTF: 1d for ADX regime filter.
- Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
- ADX > 25 = trending regime, ADX < 20 = ranging regime (with hysteresis).
- In trending regime (ADX > 25): 
    * Long when Bull Power > 0 and rising (Bull Power > prev Bull Power) with volume spike
    * Short when Bear Power > 0 and rising (Bear Power > prev Bear Power) with volume spike
- In ranging regime (ADX < 20):
    * Long when Bull Power crosses above 0 with volume spike (mean reversion from oversold)
    * Short when Bear Power crosses above 0 with volume spike (mean reversion from overbought)
- Volume spike: current volume > 1.5 * 20-period volume MA
- Exit: Opposite Elder Ray signal or volume spike in opposite direction
- Works in bull via buying strength in uptrend, in bear via selling strength in downtrend.
- Works in ranging markets via mean reversion at Elder Ray extremes.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_elder_ray(high, low, close, ema_len=13):
    """Calculate Elder Ray Bull Power and Bear Power"""
    ema = pd.Series(close).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    bull_power = high - ema
    bear_power = ema - low
    return bull_power, bear_power, ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Elder Ray on 6h data
    bull_power, bear_power, ema_13 = calculate_elder_ray(high, low, close, 13)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for ADX and 6h bars for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        bull_curr = bull_power[i]
        bear_curr = bear_power[i]
        bull_prev = bull_power[i-1] if i > 0 else 0
        bear_prev = bear_power[i-1] if i > 0 else 0
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Determine regime: ADX > 25 = trending, ADX < 20 = ranging (with hysteresis)
            if adx_val > 25:
                # Trending regime: follow Elder Ray strength
                if bull_curr > 0 and bull_curr > bull_prev and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif bear_curr > 0 and bear_curr > bear_prev and vol_spike:
                    signals[i] = -0.25
                    position = -1
            elif adx_val < 20:
                # Ranging regime: mean reversion at Elder Ray extremes
                if bull_curr > 0 and bull_prev <= 0 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif bear_curr > 0 and bear_prev <= 0 and vol_spike:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bear Power becomes positive (shift to bearish) or volume spike in opposite direction
            if bear_curr > 0 or (vol_spike and bear_curr > bull_curr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power becomes positive (shift to bullish) or volume spike in opposite direction
            if bull_curr > 0 or (vol_spike and bull_curr > bear_curr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0