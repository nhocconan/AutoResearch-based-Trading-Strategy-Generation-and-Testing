#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1d EMA34 trend filter and volume confirmation.
- Long when price breaks above Donchian(20) upper band AND 1d close > EMA34 (bullish regime)
- Short when price breaks below Donchian(20) lower band AND 1d close < EMA34 (bearish regime)
- Volume confirmation: current volume > 1.5 * 20-period average volume
- Fixed position size of 0.25 to balance return and drawdown
- Exit on opposite Donchian breakout or trend regime change
- Uses 12h primary with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Donchian provides objective breakout levels; EMA34 filter avoids chop and confirms trend
- Volume confirmation ensures breakouts have conviction, reducing false signals
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
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Trend filter: bullish if close > EMA34, bearish if close < EMA34
    bullish_regime = close > ema_34_1d_aligned
    bearish_regime = close < ema_34_1d_aligned
    
    # Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper band AND bullish regime AND volume confirmation
            if close[i] > upper[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = position_size
                position = 1
            # Short: break below lower band AND bearish regime AND volume confirmation
            elif close[i] < lower[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -position_size
                position = -1
        elif position == 1:
            # Long exit: break below lower band OR regime turns bearish
            if close[i] < lower[i] or bearish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        elif position == -1:
            # Short exit: break above upper band OR regime turns bullish
            if close[i] > upper[i] or bullish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0