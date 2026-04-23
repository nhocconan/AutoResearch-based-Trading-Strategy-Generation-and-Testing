#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day ATR volatility filter and volume confirmation.
Long when price breaks above 20-period upper band, ATR(14) > 1.5x ATR(50) (high volatility regime),
and volume > 1.5x 20-period average. Short when price breaks below 20-period lower band under same conditions.
Exit when price crosses 10-period EMA or ATR drops below 1.0x ATR(50) (low volatility).
Designed to capture strong breakouts in high volatility regimes while avoiding choppy markets.
Works in both bull and bear markets by requiring volatility expansion and clear price breaks.
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
    
    # Load 1-day data for ATR - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1-day ATR (14 and 50 period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR calculations
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF ATR to lower timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    # 4-hour Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-period EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema10[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr14_val = atr14_aligned[i]
        atr50_val = atr50_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema10_val = ema10[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian band, high volatility regime, volume confirmation
            if (close_val > donchian_high_val and 
                atr14_val > 1.5 * atr50_val and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band, high volatility regime, volume confirmation
            elif (close_val < donchian_low_val and 
                  atr14_val > 1.5 * atr50_val and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below 10-period EMA OR volatility contraction
                if (close_val < ema10_val) or (atr14_val < 1.0 * atr50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above 10-period EMA OR volatility contraction
                if (close_val > ema10_val) or (atr14_val < 1.0 * atr50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_1dATR_Volume_Breakout"
timeframe = "4h"
leverage = 1.0