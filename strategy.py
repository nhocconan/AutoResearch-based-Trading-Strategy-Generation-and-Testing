#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_trend_v1
# Strategy: 4h Donchian breakout with 1d EMA trend and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Breakouts above/below 4h Donchian channels (20-period) with volume > 1.5x 20-period average
# and aligned with 1d EMA50 trend capture sustained moves in both bull and bear markets.
# Volume and trend filters reduce false breakouts. Target 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR for stop (not used in signal but for volatility context)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: close vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Price relative to Donchian channels
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Entry conditions
        # Long: Donchian breakout up AND uptrend AND volume confirmation
        if breakout_up and uptrend and vol_confirm and position != 1:
            # Additional check: ensure we didn't just breakout in previous bar
            if i == 50 or close[i-1] <= donchian_high[i-1]:
                position = 1
                signals[i] = 0.25
        # Short: Donchian breakout down AND downtrend AND volume confirmation
        elif breakout_down and downtrend and vol_confirm and position != -1:
            # Additional check: ensure we didn't just breakout in previous bar
            if i == 50 or close[i-1] >= donchian_low[i-1]:
                position = -1
                signals[i] = -0.25
        # Exit: Price returns to the middle of the Donchian channel (mean reversion)
        elif position == 1 and close[i] < (donchian_high[i] + donchian_low[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (donchian_high[i] + donchian_low[i]) / 2:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals