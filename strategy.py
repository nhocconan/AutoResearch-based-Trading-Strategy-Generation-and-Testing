#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d HMA21 trend filter and volume confirmation.
- Long when price breaks above Donchian upper (20) AND close > 1d HMA21 (bullish trend)
- Short when price breaks below Donchian lower (20) AND close < 1d HMA21 (bearish trend)
- Volume must be > 2.0 * ATR(14) (high-volume breakout confirmation)
- Exit on trend reversal (close crosses opposite 1d HMA21) for faster mean reversion in chop
- Uses 4h primary timeframe with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Donchian channels provide robust breakout levels that work in trending markets
- 1d HMA21 ensures alignment with longer-term trend to avoid whipsaws in ranging/bear markets
- ATR-scaled volume filter adapts to changing volatility, reducing false breakouts
- Designed for BTC/ETH with edge in bull markets (breakout continuation) and bear markets (avoiding false breakouts via trend filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) using previous period (no look-ahead)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    lookback = 20
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Get 1d data ONCE before loop for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA21
    close_1d = df_1d['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    hma_21_1d = np.full_like(close_1d, np.nan)
    if len(wma_half) >= half_len and len(wma_full) >= 21:
        hma_21_1d[20:] = wma(2 * wma_half[half_len-1:] - wma_full, sqrt_len)
    
    # Align 1d HMA21 to 4h timeframe
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate ATR(14) for dynamic volume threshold
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 2.0 * ATR (high-volume breakout)
    vol_threshold = 2.0 * atr
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 21, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, trend up (close > HMA21), volume confirmation
            if close[i] > donchian_upper[i] and close[i] > hma_21_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, trend down (close < HMA21), volume confirmation
            elif close[i] < donchian_lower[i] and close[i] < hma_21_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below 1d HMA21 (trend reversal)
            if close[i] < hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above 1d HMA21 (trend reversal)
            if close[i] > hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dHMA21_ATRVolConfirm_v1"
timeframe = "4h"
leverage = 1.0