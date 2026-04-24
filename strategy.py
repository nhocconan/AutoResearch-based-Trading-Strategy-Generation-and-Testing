#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
- Long when price breaks above Donchian(20) upper band AND 1w close > EMA34 (bullish regime)
- Short when price breaks below Donchian(20) lower band AND 1w close < EMA34 (bearish regime)
- Volume confirmation: current day volume > 1.5 * 20-day average volume
- Fixed position size: 0.25 (25% of capital) to balance risk and return
- Exit on opposite Donchian breakout or trend regime change (1w close/EMA34 crossover)
- Uses 1d primary with 1w HTF targeting 30-100 total trades over 4 years (7-25/year)
- Donchian provides objective breakout levels; 1w EMA34 filter avoids counter-trend trades in bear markets
- Volume confirmation ensures breakouts have institutional participation
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
    
    # Donchian(20) channels
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 20-day average volume for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe (waits for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Regime: bullish if 1w close > EMA34, bearish if 1w close < EMA34
    bullish_regime = df_1w['close'].values > ema_34_1w
    bearish_regime = df_1w['close'].values < ema_34_1w
    bullish_regime_aligned = align_htf_to_ltf(prices, df_1w, bullish_regime)
    bearish_regime_aligned = align_htf_to_ltf(prices, df_1w, bearish_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(bullish_regime_aligned[i]) or np.isnan(bearish_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: break above upper band AND bullish regime AND volume confirmation
            if close[i] > upper[i] and bullish_regime_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band AND bearish regime AND volume confirmation
            elif close[i] < lower[i] and bearish_regime_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below lower band OR regime turns bearish
            if close[i] < lower[i] or bearish_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above upper band OR regime turns bullish
            if close[i] > upper[i] or bullish_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0