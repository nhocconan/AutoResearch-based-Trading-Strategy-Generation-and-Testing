#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long when close > upper Donchian(20) AND ATR(14) > 1.5 * ATR(50) (high volatility regime) AND volume > 1.3 * 20-period average
- Short when close < lower Donchian(20) AND ATR(14) > 1.5 * ATR(50) AND volume > 1.3 * 20-period average
- Exit when price crosses opposite Donchian band OR ATR(14) < ATR(50) (low volatility regime)
- Uses 12h primary with 1d HTF for ATR regime filter to avoid whipsaws in low volatility markets
- Donchian channels provide clear breakout levels; ATR filter ensures trades only in high conviction volatile markets; volume confirms breakout strength
- Designed to work in both bull (strong upward breaks) and bear (strong downward breaks) markets with volatility filter
- Signal size: 0.30 discrete levels to balance profit potential and risk management
- Target: 50-150 total trades over 4 years (12-37/year)
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
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_donchian = rolling_max(high, 20)
    lower_donchian = rolling_min(low, 20)
    
    # Calculate ATR (14-period) for volatility regime filter
    def calculate_atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14 = calculate_atr(high, low, close, 14)
    atr_50 = calculate_atr(high, low, close, 50)
    
    # Calculate 1d ATR for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d_14 = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_50 = calculate_atr(high_1d, low_1d, close_1d, 50)
    
    # ATR regime filter: high volatility when short ATR > long ATR * threshold
    high_vol_regime = atr_1d_14 > (atr_1d_50 * 1.5)
    low_vol_regime = atr_1d_14 < atr_1d_50
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20) + 1  # Need Donchian, ATR50, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or 
            np.isnan(high_vol_regime_aligned[i]) or np.isnan(low_vol_regime_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper Donchian AND high volatility regime AND volume confirmation
            if close[i] > upper_donchian[i] and high_vol_regime_aligned[i] and volume_confirm[i]:
                signals[i] = 0.30
                position = 1
            # Short: break below lower Donchian AND high volatility regime AND volume confirmation
            elif close[i] < lower_donchian[i] and high_vol_regime_aligned[i] and volume_confirm[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: break below lower Donchian OR low volatility regime
            if close[i] < lower_donchian[i] or low_vol_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: break above upper Donchian OR low volatility regime
            if close[i] > upper_donchian[i] or low_vol_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0