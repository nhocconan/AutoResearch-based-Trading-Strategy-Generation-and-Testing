#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_EMA_Trend_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with zero-lag EMA trend filter and ATR-based regime.
Long when Bull Power > 0, zero-lag EMA rising, and ATR ratio < 0.8 (low volatility regime).
Short when Bear Power < 0, zero-lag EMA falling, and ATR ratio < 0.8.
Elder Ray measures trend strength relative to EMA13; zero-lag EMA reduces lag for timely entries.
ATR regime filter avoids choppy markets. Designed for 12-37 trades/year by requiring confluence.
Works in bull/bear via trend direction: takes longs in uptrend, shorts in downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for HTF trend and regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate ATR(14) and ATR(50) for regime filter (low volatility)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / (atr_50 + 1e-10)  # avoid division by zero
    
    # Calculate EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Zero-lag EMA (21) for timely trend direction
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_lag = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    zero_lag_ema = 2 * ema_21 - ema21_lag  # reduces lag
    zero_lag_ema_rising = zero_lag_ema > np.roll(zero_lag_ema, 1)
    zero_lag_ema_falling = zero_lag_ema < np.roll(zero_lag_ema, 1)
    # Handle first bar
    zero_lag_ema_rising[0] = False
    zero_lag_ema_falling[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for ATR50, 21 for zero-lag EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(ema_13[i]) or np.isnan(zero_lag_ema[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade in low volatility (ATR ratio < 0.8)
        low_vol_regime = atr_ratio[i] < 0.8
        
        if htf_trend[i] == 1 and low_vol_regime:  # Uptrend on 1d + low vol
            # Long when Bull Power > 0 and zero-lag EMA rising
            if bull_power[i] > 0 and zero_lag_ema_rising[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if Bull Power turns negative (momentum loss)
            elif position == 1 and bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1 and low_vol_regime:  # Downtrend on 1d + low vol
            # Short when Bear Power < 0 and zero-lag EMA falling
            if bear_power[i] < 0 and zero_lag_ema_falling[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if Bear Power turns positive (momentum loss)
            elif position == -1 and bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # In high volatility or no clear trend, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_ZeroLag_EMA_Trend_v1"
timeframe = "6h"
leverage = 1.0