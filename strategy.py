#!/usr/bin/env python3
"""
Experiment #416: 12h Donchian Breakout + 1d Volume Spike + ATR Regime Filter

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe, confirmed by 1d volume spike (>2x average) 
and filtered by ATR-based regime (trending when ATR(14) > ATR(50)), captures strong momentum 
moves in both bull and bear markets. The 12h timeframe minimizes fee drag while Donchian 
breakouts provide objective entry/exit levels. Volume confirmation ensures institutional 
participation, and ATR regime filter avoids choppy markets. Targets 12-37 trades/year on 12h 
timeframe (50-150 total over 4 years) to minimize fee drag while targeting high-probability 
breakouts with clear stoploss and profit-taking at opposite Donchian band.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike and ATR regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # Calculate ATR(14) and ATR(50) on 1d for regime filter
    if len(df_1d) >= 50:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], 
                          abs(high_1d[i] - close_1d[i-1]), 
                          abs(low_1d[i] - close_1d[i-1]))
        
        # ATR(14) and ATR(50)
        atr_14_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        atr_50_1d = pd.Series(tr_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        
        # Regime: trending when ATR(14) > ATR(50) (increasing volatility)
        atr_regime_1d = atr_14_1d > atr_50_1d
        atr_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_1d.astype(np.float64))
    else:
        atr_regime_1d_aligned = np.full(n, 1.0)  # Default to trending if insufficient data
    
    # === 12h Indicators ===
    # Calculate Donchian channels (20-period) on 12h using HTF data
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) >= 20:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        
        # Donchian channels: upper = max(high, 20), lower = min(low, 20)
        donchian_upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        donchian_lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        
        # Align to lower timeframe
        donchian_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
        donchian_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    else:
        donchian_upper_12h_aligned = np.full(n, np.nan)
        donchian_lower_12h_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper_12h_aligned[i]) or np.isnan(donchian_lower_12h_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(atr_regime_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when ATR(14) > ATR(50) (trending market) ---
        is_trending = atr_regime_1d_aligned[i] > 0.5
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic ---
        if in_position:
            # Exit when price touches opposite Donchian band
            if position_side > 0:  # Long position
                if low[i] <= donchian_lower_12h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if high[i] >= donchian_upper_12h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band with volume and trend
        long_condition = (
            close[i] > donchian_upper_12h_aligned[i] and 
            volume_spike and 
            is_trending
        )
        
        # Short: Price breaks below Donchian lower band with volume and trend
        short_condition = (
            close[i] < donchian_lower_12h_aligned[i] and 
            volume_spike and 
            is_trending
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals