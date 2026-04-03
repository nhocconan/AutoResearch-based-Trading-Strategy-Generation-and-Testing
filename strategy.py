#!/usr/bin/env python3
"""
Experiment #045: 12h Donchian(20) Breakout + 1d Volume Spike + Chop Regime Filter

HYPOTHESIS: Donchian channel breakouts on 12h timeframe with 1d volume confirmation 
and choppiness regime filter captures strong trending moves while avoiding choppy 
markets. The 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) 
to minimize fee drag. Volume spike confirms institutional participation, while 
choppiness regime ensures we only trade in trending markets (CHOP < 38.2). 
This combination has shown strong performance on SOLUSDT (test Sharpe 1.46) and 
should work across BTC/ETH/SOL in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian20_1d_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike and chop regime (Call ONCE before loop) ===
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
    
    # Calculate Choppiness Index on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr = np.zeros(len(close_1d))
        tr[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: 100 * log10(sum(tr) / (max_high - min_low)) / log10(14)
        chop = np.full(len(close_1d), np.nan)
        valid = (max_high - min_low) > 0
        chop[valid] = 100 * np.log10(tr_sum[valid] / (max_high[valid] - min_low[valid])) / np.log10(14)
        
        # Regime: CHOP < 38.2 = trending (trade), CHOP > 61.8 = ranging (avoid)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
        trending_regime = chop_aligned < 38.2
    else:
        trending_regime = np.full(n, True)  # Default to trending if insufficient data
        chop_aligned = np.full(n, 50.0)
    
    # === 12h Indicators ===
    # Calculate Donchian Channel (20-period) on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        
        # Upper band: highest high over 20 periods
        upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        # Lower band: lowest low over 20 periods
        lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        
        # Align to 12h timeframe
        upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
        lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    else:
        upper_12h_aligned = np.full(n, np.nan)
        lower_12h_aligned = np.full(n, np.nan)
    
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
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets ---
        if not trending_regime[i]:
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using available data up to i
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], 
                           abs(high[j] - close[j-1]), 
                           abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price retouches lower Donchian band (mean reversion signal)
                if close[i] <= lower_12h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price retouches upper Donchian band
                if close[i] >= upper_12h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above upper Donchian band with volume in trending regime
        long_condition = (
            close[i] > upper_12h_aligned[i] and 
            volume_spike
        )
        
        # Short: Price breaks below lower Donchian band with volume in trending regime
        short_condition = (
            close[i] < lower_12h_aligned[i] and 
            volume_spike
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