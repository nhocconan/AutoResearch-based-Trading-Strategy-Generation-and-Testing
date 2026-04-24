#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Long when price breaks above 20-period Donchian upper band AND close > 1w EMA50 (bullish trend)
- Short when price breaks below 20-period Donchian lower band AND close < 1w EMA50 (bearish trend)
- Volume must be > 1.5 * ATR(14) * close (volatility-adjusted volume filter)
- Exit on trend reversal (close crosses opposite 1w EMA50)
- Uses 1d primary timeframe with 1w HTF to target 30-100 trades over 4 years (7-25/year)
- Donchian channels provide clear breakout levels that work in trending markets
- 1w EMA50 ensures alignment with longer-term trend to avoid whipsaws in ranging/bear markets
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
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for dynamic volume threshold
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 1.5 * ATR * close (volatility-adjusted)
    vol_threshold = 1.5 * atr * close
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band, trend up (close > EMA50), volume confirmation
            if close[i] > donchian_upper[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, trend down (close < EMA50), volume confirmation
            elif close[i] < donchian_lower[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below 1w EMA50 (trend reversal)
            if close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above 1w EMA50 (trend reversal)
            if close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_ATRVolConfirm_v1"
timeframe = "1d"
leverage = 1.0