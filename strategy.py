#!/usr/bin/env python3
"""
6h_12h_Donchian_Breakout_Volume_Regime
Hypothesis: On 6h timeframe, use 12h Donchian channel breakouts with volume confirmation and volatility regime filter.
In low volatility (12h ATR < 50th percentile): require stronger breakout (2x ATR) and higher volume (2x avg).
In high volatility: trade standard breakout with volume confirmation.
Designed to work in both bull and bear by adapting to volatility conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_Donchian_Breakout_Volume_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H DATA FOR DONCHIAN CHANNEL AND REGIME ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Donchian channel (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h ATR for volatility regime and breakout filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 12h ATR percentile (50-period lookback) - regime filter
    atr_series = pd.Series(atr_12h)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    atr_regime = align_htf_to_ltf(prices, df_12h, atr_percentile)  # < 0.5 = low vol regime
    
    # Align Donchian channels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === 6H DATA FOR VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Need enough lookback for indicators
        # Skip if not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or np.isnan(atr_regime[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime: low volatility (< 50th percentile) = stricter entry
        low_vol_regime = atr_regime[i] < 0.5
        
        if low_vol_regime:
            # LOW VOL: Require stronger breakout and higher volume
            breakout_threshold = atr_12h_aligned[i] * 2.0
            volume_threshold = 2.0
            
            bullish_breakout = (close[i] > donchian_high_aligned[i] + breakout_threshold) and (vol_ratio[i] > volume_threshold)
            bearish_breakout = (close[i] < donchian_low_aligned[i] - breakout_threshold) and (vol_ratio[i] > volume_threshold)
            
            bullish_setup = bullish_breakout
            bearish_setup = bearish_breakout
            
            # Exit when price returns to Donchian channel
            exit_long = close[i] <= donchian_high_aligned[i]
            exit_short = close[i] >= donchian_low_aligned[i]
            
        else:
            # HIGH VOL: Standard breakout with volume confirmation
            breakout_threshold = atr_12h_aligned[i] * 0.5
            volume_threshold = 1.5
            
            bullish_breakout = (close[i] > donchian_high_aligned[i] + breakout_threshold) and (vol_ratio[i] > volume_threshold)
            bearish_breakout = (close[i] < donchian_low_aligned[i] - breakout_threshold) and (vol_ratio[i] > volume_threshold)
            
            bullish_setup = bullish_breakout
            bearish_setup = bearish_breakout
            
            # Exit when price returns to opposite Donchian band (mean reversion tendency)
            exit_long = close[i] <= donchian_low_aligned[i]
            exit_short = close[i] >= donchian_high_aligned[i]
        
        # Execute trades
        if bullish_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals