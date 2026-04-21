#!/usr/bin/env python3
"""
4h_KAMA_Regime_Filter_DonchianExit
Hypothesis: 4h KAMA (adaptive trend) determines market regime (trending vs choppy).
In trending regime (price > KAMA), enter long on Donchian(20) breakout with volume spike.
In choppy regime (price < KAMA), enter short on Donchian(20) breakdown with volume spike.
Volume confirmation (2.0x average) reduces false breakouts. ATR(14) stoploss (2.0x).
Uses discrete sizing (0.25) to minimize fee churn. Designed to work in both bull and bear markets
by adapting to regime changes. Target: 20-40 trades/year per symbol (<150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 4h KAMA (adaptive trend) for regime filter ===
    close = prices['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    fastest = 2.0 / (2 + 1)   # EMA(2)
    slowest = 2.0 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])
            or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Regime: trending if price > KAMA, choppy if price < KAMA
            if price > kama[i]:  # trending regime
                # Long on Donchian breakout with volume
                if price > donchian_high[i] and volume_confirmed:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            else:  # choppy regime
                # Short on Donchian breakdown with volume
                if price < donchian_low[i] and volume_confirmed:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime change exit
            elif price < kama[i]:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at Donchian low
            elif price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime change exit
            elif price > kama[i]:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at Donchian high
            elif price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Regime_Filter_DonchianExit"
timeframe = "4h"
leverage = 1.0