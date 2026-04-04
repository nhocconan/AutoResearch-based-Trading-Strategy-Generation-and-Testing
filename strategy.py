#!/usr/bin/env python3
"""
Experiment #2919: 6h Elder Ray + ADX Regime + Volume Confirmation
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) measures trend strength relative to EMA13.
ADX > 25 filters for trending markets, while ADX < 20 indicates ranging conditions.
In trending markets (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
In ranging markets (ADX < 20): mean revert at EMA13 extremes (price > EMA13 + 0.5*ATR for short, < EMA13 - 0.5*ATR for long).
Volume confirmation (>1.5x average) ensures participation. 6h timeframe balances responsiveness and noise.
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2919_6h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 6h Indicators: ATR(14) for volatility ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === 6h Indicators: ADX(14) for regime detection ===
    # +DM and -DM calculation
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Smoothed +DM, -DM, TR
    atr_for_dx = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_for_dx
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_for_dx
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = max(50, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        volume_confirmed = vol_ratio[i] > 1.5
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
            
        # Regime detection: ADX > 25 = trending, ADX < 20 = ranging
        if adx[i] > 25:  # Trending market
            # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
            bull_power = high[i] - ema13[i]
            bear_power = low[i] - ema13[i]
            
            # Long when Bull Power positive and rising (strong uptrend)
            if bull_power > 0 and i > warmup and bull_power > (high[i-1] - ema13[i-1]):
                signals[i] = SIZE
            # Short when Bear Power negative and falling (strong downtrend)
            elif bear_power < 0 and i > warmup and bear_power < (low[i-1] - ema13[i-1]):
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        elif adx[i] < 20:  # Ranging market
            # Mean reversion at EMA13 extremes
            if price > ema13[i] + 0.5 * atr[i]:
                signals[i] = -SIZE  # Short at upper extreme
            elif price < ema13[i] - 0.5 * atr[i]:
                signals[i] = SIZE   # Long at lower extreme
            else:
                signals[i] = 0.0
        else:  # Transition regime (20 <= ADX <= 25) - stay flat
            signals[i] = 0.0
    
    return signals