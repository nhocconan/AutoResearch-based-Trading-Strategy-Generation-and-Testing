#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian channel breakout (20) with 1w EMA34 trend filter and ATR volume confirmation.
- Long when price breaks above Donchian upper (20) AND close > 1w EMA34 (bullish trend)
- Short when price breaks below Donchian lower (20) AND close < 1w EMA34 (bearish trend)
- Volume must be > 1.5 * ATR(14) (volatility-adjusted volume filter to avoid fakeouts)
- Exit on opposite Donchian breakout or trend reversal (close crosses 1w EMA34)
- Uses 1d primary timeframe with 1w HTF to target 30-100 trades over 4 years (7-25/year)
- Donchian channels provide robust structure that works in both trending and ranging markets
- 1w EMA34 ensures alignment with long-term trend to avoid whipsaws in bear markets
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
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for dynamic volume threshold
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 1.5 * ATR (volatility-adjusted)
    vol_threshold = 1.5 * atr
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, trend up (close > EMA34), volume confirmation
            if close[i] > donchian_upper[i] and close[i] > ema_34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, trend down (close < EMA34), volume confirmation
            elif close[i] < donchian_lower[i] and close[i] < ema_34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR trend reversal (close < EMA34)
            if close[i] < donchian_lower[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR trend reversal (close > EMA34)
            if close[i] > donchian_upper[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_ATRVolConfirm_v1"
timeframe = "1d"
leverage = 1.0