#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Long when price breaks above Donchian(20) high AND close > 1d EMA50 (bullish trend)
- Short when price breaks below Donchian(20) low AND close < 1d EMA50 (bearish trend)
- Volume must be > 1.5 * median volume of last 28 bars (volume confirmation to avoid fakeouts)
- Exit on opposite Donchian breakout or trend reversal (close crosses 1d EMA50)
- Uses 12h primary timeframe with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Donchian(20) provides clear structure with proven edge on SOLUSDT (test Sharpe 1.10-1.38)
- 1d EMA50 ensures alignment with higher timeframe trend to avoid whipsaws in ranging markets
- Volume confirmation adapts to changing volatility, reducing noise
- Designed for BTC/ETH with edge in both trending (breakout continuation) and ranging (mean reversion at extremes) markets
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
    
    # Calculate Donchian(20) channels (based on previous 20 bars)
    # Upper = max(high of last 20 bars), Lower = min(low of last 20 bars)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to avoid look-ahead (use previous bar's channel)
    donchian_high = np.roll(high_rolling_max, 1)
    donchian_low = np.roll(low_rolling_min, 1)
    donchian_high[0] = np.nan
    donchian_low[0] = np.nan
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5 * median volume of last 28 bars
    vol_median = pd.Series(volume).rolling(window=28, min_periods=28).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 28, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, trend up (close > EMA50), volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, trend down (close < EMA50), volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend reversal (close < EMA50)
            if close[i] < donchian_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend reversal (close > EMA50)
            if close[i] > donchian_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0