#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1w EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND close < 1w EMA50 AND volume > 1.5x 20-period average.
Exit when price crosses Donchian midpoint (mean of upper and lower band).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
The 1w EMA50 provides a robust long-term trend filter that adapts to bull/bear markets.
Volume confirmation at 1.5x ensures only high-momentum breakouts are taken, reducing false signals.
Donchian channels from 12h provide clear structure with proven edge on multiple symbols.
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
    
    # Load 12h data for Donchian channels - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian levels to primary timeframe (15m)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND close > 1w EMA50 AND volume spike
            if (price > donchian_upper_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND close < 1w EMA50 AND volume spike
            elif (price < donchian_lower_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian midpoint
            if position == 1 and price < donchian_mid_aligned[i]:
                exit_signal = True
            elif position == -1 and price > donchian_mid_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "15m"
leverage = 1.0