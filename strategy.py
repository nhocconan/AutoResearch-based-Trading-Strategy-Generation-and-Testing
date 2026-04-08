#!/usr/bin/env python3
# 4h_donchian_volume_atr_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based regime filter.
# Long: Close breaks above upper Donchian(20) AND volume > 1.5 * volume MA(20) AND ATR(14) > ATR(50) (volatile regime)
# Short: Close breaks below lower Donchian(20) AND volume > 1.5 * volume MA(20) AND ATR(14) > ATR(50) (volatile regime)
# Exit: Opposite Donchian breakout or ATR(14) < ATR(50) * 0.8 (low volatility exit)
# Uses 4h primary timeframe with 12h HTF for trend filter (EMA21) to avoid counter-trend trades.
# Target: 100-180 total trades over 4 years (25-45/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA(20) for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) and ATR(50) for regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_s = pd.Series(tr)
    atr14 = tr_s.rolling(window=14, min_periods=14).mean().values
    atr50 = tr_s.rolling(window=50, min_periods=50).mean().values
    
    # Get 12h data for HTF trend filter (EMA21)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    close_12h_s = pd.Series(close_12h)
    ema21_12h = close_12h_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i]) or
            np.isnan(ema21_12h_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period volume MA
        volume_confirm = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: ATR(14) > ATR(50) (volatile enough to trade)
        volatile_regime = atr14[i] > atr50[i]
        
        # HTF trend filter: only trade in direction of 12h EMA21
        bullish_htf = close[i] > ema21_12h_aligned[i]
        bearish_htf = close[i] < ema21_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Opposite Donchian breakout (close below lower Donchian)
            # 2. Low volatility exit (ATR(14) < ATR(50) * 0.8)
            if (close[i] < donchian_lower[i]) or (atr14[i] < atr50[i] * 0.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Opposite Donchian breakout (close above upper Donchian)
            # 2. Low volatility exit (ATR(14) < ATR(50) * 0.8)
            if (close[i] > donchian_upper[i]) or (atr14[i] < atr50[i] * 0.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Close breaks above upper Donchian AND volume confirmation AND volatile regime AND bullish HTF
            if (close[i] > donchian_upper[i]) and volume_confirm and volatile_regime and bullish_htf:
                position = 1
                signals[i] = 0.25
            # Short entry: Close breaks below lower Donchian AND volume confirmation AND volatile regime AND bearish HTF
            elif (close[i] < donchian_lower[i]) and volume_confirm and volatile_regime and bearish_htf:
                position = -1
                signals[i] = -0.25
    
    return signals