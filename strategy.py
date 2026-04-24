#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) Breakout with 1w EMA34 Trend Filter and Volume Spike.
- Donchian(20) levels from daily chart capture intermediate-term momentum breakouts.
- 1w EMA34 provides higher-timeframe trend filter to align with weekly momentum and reduce counter-trend trades.
- Volume spike (>2x 20-period average volume) confirms breakout validity and reduces false signals.
- Discrete position sizing (0.30) balances return potential with fee minimization.
- Target trades: 30-100 total over 4 years (7-25/year) to avoid fee drag on 1d timeframe.
- Works in bull/bear markets via 1w trend filter and volatility-based volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data ONCE before loop for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper (20-period high) and lower (20-period low)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (using previous completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian upper with volume spike and above 1w EMA34 (bullish higher-timeframe trend)
            if close[i] > donchian_upper_aligned[i] and volume_spike[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: break below Donchian lower with volume spike and below 1w EMA34 (bearish higher-timeframe trend)
            elif close[i] < donchian_lower_aligned[i] and volume_spike[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian lower OR below 1w EMA34 (trend change)
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price closes above Donchian upper OR above 1w EMA34 (trend change)
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0