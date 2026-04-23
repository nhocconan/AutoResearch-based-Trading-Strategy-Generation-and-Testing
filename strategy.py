#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ATR-based volatility filter and volume confirmation
- Uses Donchian(20) channels for breakout signals
- ATR(14) from 1d timeframe as volatility filter: only trade when ATR ratio > 0.8 (avoid low volatility)
- Volume confirmation (> 1.3x 20-period average) ensures momentum behind breakouts
- Trend filter using 4h EMA(50): only long when price > EMA, short when price < EMA
- Designed for 4h timeframe targeting 20-40 trades/year (80-160 over 4 years)
- Works in both bull and bear markets by combining breakout with volatility regime filter
- ATR-based volatility filter adapts to changing market conditions (more trades in high vol, fewer in low vol)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_ma
    lower_channel = low_ma
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar TR
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar TR (no previous close)
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First bar TR (no previous close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR average) for volatility regime
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / np.where(atr_ma_50 > 0, atr_ma_50, 1)  # Avoid division by zero
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50, 20)  # Donchian, ATR, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(ema_50[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when volatility is elevated (ATR ratio > 0.8)
        volatility_filter = atr_ratio_aligned[i] > 0.8
        
        # Determine breakout conditions
        price_above_upper = close[i] > upper_channel[i]
        price_below_lower = close[i] < lower_channel[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
        if position == 0:
            # Long conditions: price breaks above upper channel, uptrend, volume spike, volatility filter
            long_signal = (price_above_upper and 
                          uptrend and
                          volume[i] > 1.3 * vol_ma[i] and
                          volatility_filter)
            
            # Short conditions: price breaks below lower channel, downtrend, volume spike, volatility filter
            short_signal = (price_below_lower and 
                           downtrend and
                           volume[i] > 1.3 * vol_ma[i] and
                           volatility_filter)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below lower channel or trend turns down
                if (price_below_lower or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above upper channel or trend turns up
                if (price_above_upper or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_VolumeFilter_EMA50Trend"
timeframe = "4h"
leverage = 1.0