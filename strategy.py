#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout for trend direction
# and 1d ATR for volatility filtering + volume confirmation
# - Uses 1d HTF for Donchian(20): breakout above/below 20-period high/low determines trend
# - Uses 1d HTF for ATR(14): volatility filter to avoid low-volatility false breakouts
# - Volume confirmation: current 12h volume > 1.5x 20-period average to ensure strong participation
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets by trading breakouts in the direction of the higher timeframe trend

name = "12h_1d_donchian_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20 periods)
    period20_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 12h timeframe (wait for completed HTF bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, period20_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, period20_low)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0 or atr_14_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: ATR > 0.5 * 20-period ATR average (avoid low volatility)
        atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=1).mean().values
        volatility_filter = atr_14_aligned[i] > 0.5 * atr_ma_20[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: price retouches Donchian low or trend change
            if close[i] <= donchian_low_aligned[i] or close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: price retouches Donchian high or trend change
            if close[i] >= donchian_high_aligned[i] or close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic: Donchian breakout with volume and volatility confirmation
            if volume_confirmed and volatility_filter:
                if close[i] > donchian_high_aligned[i]:
                    # Breakout above Donchian high: long
                    position = 1
                    signals[i] = position_size
                elif close[i] < donchian_low_aligned[i]:
                    # Breakout below Donchian low: short
                    position = -1
                    signals[i] = -position_size
    
    return signals