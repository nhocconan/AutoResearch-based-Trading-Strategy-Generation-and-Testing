#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATR_Volume_Regime_v1
Hypothesis: Trade Donchian(20) breakouts with ATR-based position sizing (0.25) and volume confirmation (>1.5x 20-bar average). Only trade when choppiness regime indicates trend (CHOP < 38.2) to avoid whipsaws in ranging markets. Uses 1d EMA50 as HTF trend filter for alignment. Designed for low trade frequency (<25/year) to minimize fee drag while capturing strong trending moves. Works in both bull and bear regimes by following HTF trend direction.
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
    
    # Get 1d data for HTF trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14) for volatility filtering
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    # Calculate 4h Choppiness Index (CHOP) for regime filter
    def calculate_chop(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TR over window
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Highest high and lowest low over window
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        # Choppiness Index
        chop = np.zeros(len(close))
        for i in range(len(close)):
            if tr_sum[i] > 0 and hh[i] > ll[i]:
                chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(window)
            else:
                chop[i] = 50.0  # neutral value
        return chop
    
    chop = calculate_chop(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(50, 20, 14, 20)  # EMA50 needs 50, Donchian needs 20, ATR needs 14, vol needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        df_1d_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(df_1d_close_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        htf_1d_bullish = df_1d_close_aligned[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = df_1d_close_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: moderate spike (vol_ratio > 1.5)
        volume_confirmed = vol_ratio[i] > 1.5
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        regime_filter = chop[i] < 38.2
        
        if position == 0:
            # Long setup: price breaks above Donchian high + 1d uptrend + volume confirmation + trending regime
            long_setup = (close[i] > donchian_high[i]) and htf_1d_bullish and volume_confirmed and regime_filter
            
            # Short setup: price breaks below Donchian low + 1d downtrend + volume confirmation + trending regime
            short_setup = (close[i] < donchian_low[i]) and htf_1d_bearish and volume_confirmed and regime_filter
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Donchian low (opposite band) OR 1d trend turns bearish
            if (close[i] <= donchian_low[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian high (opposite band) OR 1d trend turns bullish
            if (close[i] >= donchian_high[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_ATR_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0