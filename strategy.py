#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation, and chop regime filter
# Long when price breaks above Donchian(20) high, price > 1d EMA34, volume > 1.5x 20-bar average, and CHOP > 61.8 (range)
# Short when price breaks below Donchian(20) low, price < 1d EMA34, volume > 1.5x 20-bar average, and CHOP > 61.8 (range)
# Uses 1d EMA for higher timeframe trend alignment (matches experiment HTF)
# Volume spike confirms breakout strength
# Chop regime filter ensures we only trade in ranging markets where mean reversion works
# Designed for low trade frequency (19-50/year on 4h) to avoid fee drag
# Works in bull (breakouts above rising EMA) and bear (breakdowns below falling EMA) via range trading

name = "4h_Donchian20_Volume_1dEMA34_Chop_Range_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 4h
    # Donchian high = rolling max of high over 20 periods
    # Donchian low = rolling min of low over 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Calculate Choppiness Index (CHOP) on 4h
    # CHOP = 100 * log10(sum(ATR(1), n) / (max(high, n) - min(low, n))) / log10(n)
    # Where n = 14 (default period)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # First TR is just high-low
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1 / (max_high - min_low)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((max_high - min_low) > 0, chop, 50.0)  # Default to 50 when range is zero
    chop_regime = chop > 61.8  # Range regime (CHOP > 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 20, 14) + 1  # EMA(34) + Donchian(20) + CHOP(14) warmup + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian high, price > 1d EMA34, volume spike, chop regime (range)
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_1d_aligned[i] and volume_spike[i] and chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian low, price < 1d EMA34, volume spike, chop regime (range)
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike[i] and chop_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian low or price < 1d EMA34
            if (close[i] < donchian_low[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian high or price > 1d EMA34
            if (close[i] > donchian_high[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals