#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d ATR regime filter and volume spike confirmation.
- Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
- 1d ATR percentile filter: only trade when ATR(14) > 30th percentile (avoid low-vol chop).
- Volume confirmation (>1.8x 20-bar average) ensures institutional participation.
- Position size 0.25 balances profit and drawdown control.
- Works in bull/bear markets via ATR regime filter and Elder Ray's trend-following nature.
- Target trades: 80-160 total over 4 years (20-40/year) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d ATR(14) and its 30th percentile (from 50-day lookback)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    # 50-day percentile rank of ATR
    atr_percentile = pd.Series(atr_14_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    atr_30th_percentile = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_30th_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR regime filter: only trade when ATR > 30th percentile (avoid low-vol chop)
        vol_regime = atr_30th_percentile[i] > 0.30
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms and in volatile regime
            if volume_confirm and vol_regime:
                # Elder Ray bullish: strong bull power and weak bear power
                if bull_power[i] > 0 and bear_power[i] < bull_power[i]:
                    signals[i] = 0.25
                    position = 1
                # Elder Ray bearish: strong bear power and weak bull power
                elif bear_power[i] > 0 and bull_power[i] < bear_power[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: bear power exceeds bull power OR volatility drops
            if bear_power[i] >= bull_power[i] or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bull power exceeds bear power OR volatility drops
            if bull_power[i] >= bear_power[i] or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dATRRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0