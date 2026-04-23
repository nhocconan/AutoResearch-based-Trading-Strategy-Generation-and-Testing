#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
In bull markets: price > 1d EMA50 = trend up, buy Donchian(20) breakout.
In bear markets: price < 1d EMA50 = trend down, sell Donchian(20) breakout.
Volume confirmation (1.5x 20-period MA) filters false breakouts.
ATR regime filter: only trade when ATR(14) > ATR(50) (expanding volatility).
Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year per symbol.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) and ATR(50) for regime filter
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr1).rolling(window=50, min_periods=50).mean().values
    atr_regime = atr_14 > atr_50  # expanding volatility regime
    
    # Calculate 4h Donchian(20) channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels based on previous 20 periods
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 4h timeframe (previous bar values)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # need EMA50, ATR50, and vol MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA50 = uptrend, close < 1d EMA50 = downtrend
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: expanding volatility
        regime_filter = atr_regime[i]
        
        if position == 0:
            # Long: Break above Donchian high AND uptrend AND volume AND regime
            if close[i] > donchian_high_aligned[i] and trend_up and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND downtrend AND volume AND regime
            elif close[i] < donchian_low_aligned[i] and trend_down and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian level
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian low
                if close[i] < donchian_low_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian high
                if close[i] > donchian_high_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA50_Trend_Volume_ATR_Regime"
timeframe = "4h"
leverage = 1.0