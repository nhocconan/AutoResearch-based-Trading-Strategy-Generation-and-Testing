#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power (Bull/Bear) + 1d ADX regime + volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ADX regime filter.
- Elder Ray: Bull Power = high - EMA(13), Bear Power = low - EMA(13) on 6h.
- Regime filter: ADX(14) > 25 = trending (trade with Elder Ray), ADX < 20 = ranging (fade Elder Ray extremes).
- Volume confirmation: current volume > 1.5x 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying strong Bull Power in uptrend, in bear via selling strong Bear Power in downtrend, and fading extremes in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_di_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_di_smooth / atr
    minus_di = 100 * minus_di_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray on 6h: Bull Power = high - EMA(13), Bear Power = low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 20)  # EMA13 + ADX buffer + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime: ADX > 25 = trending, ADX < 20 = ranging
            if adx_aligned[i] > 25:
                # Trending regime: trade with Elder Ray power
                if bull_power[i] > 0 and bear_power[i] < 0 and volume_spike[i]:
                    # Both bulls and bears active, but check momentum
                    if bull_power[i] > abs(bear_power[i]):  # Bull power stronger
                        signals[i] = 0.25
                        position = 1
                    elif bear_power[i] < -bull_power[i]:  # Bear power stronger (more negative)
                        signals[i] = -0.25
                        position = -1
            elif adx_aligned[i] < 20:
                # Ranging regime: fade Elder Ray extremes
                if bull_power[i] > 0 and volume_spike[i]:
                    # Strong bull power in range: sell expecting reversion to mean
                    signals[i] = -0.25
                    position = -1
                elif bear_power[i] < 0 and volume_spike[i]:
                    # Strong bear power in range: buy expecting reversion to mean
                    signals[i] = 0.25
                    position = 1
        elif position == 1:
            # Long exit: bear power turns positive (bulls losing strength) or opposite signal
            if bear_power[i] >= 0 or (bull_power[i] < 0 and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bull power turns negative (bears losing strength) or opposite signal
            if bull_power[i] <= 0 or (bear_power[i] > 0 and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0