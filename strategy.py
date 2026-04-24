#!/usr/bin/env python3
"""
Hypothesis: 1h Bollinger Band breakout with 4h EMA50 trend filter and 1d ATR volume confirmation.
- Long when price breaks above BB upper (20,2) AND 4h EMA50 slope > 0 (bullish trend)
- Short when price breaks below BB lower (20,2) AND 4h EMA50 slope < 0 (bearish trend)
- Volume must be > 1.5 * 1d ATR(14) (volatility-adjusted volume filter to avoid fakeouts)
- Exit on opposite BB breakout or trend reversal (4h EMA50 slope changes sign)
- Uses 1h primary timeframe with 4h HTF for trend and 1d HTF for volume filter to target 60-150 trades over 4 years (15-37/year)
- Bollinger Bands provide dynamic support/resistance that adapts to volatility
- 4h EMA50 ensures alignment with medium-term trend to avoid whipsaws in ranging markets
- 1d ATR-scaled volume filter reduces false breakouts during low-volume periods
- Designed for BTC/ETH with edge in trending markets (breakout continuation) and avoids chop via trend filter
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
    
    # Calculate Bollinger Bands (20,2) using previous period (no look-ahead)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = (sma + bb_std * std).shift(1).values
    bb_lower = (sma - bb_std * std).shift(1).values
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 and its slope
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope_4h = np.diff(ema_50_4h, prepend=ema_50_4h[0])
    
    # Align 4h EMA50 slope to 1h timeframe
    ema_slope_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_slope_4h)
    
    # Get 1d data ONCE before loop for ATR volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ATR(14) to 1h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Dynamic volume threshold: volume > 1.5 * 1d ATR(14)
    vol_threshold = 1.5 * atr_1d_aligned
    volume_confirm = volume > vol_threshold
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_slope_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above BB upper, bullish trend (EMA50 slope > 0), volume confirmation
            if close[i] > bb_upper[i] and ema_slope_4h_aligned[i] > 0 and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below BB lower, bearish trend (EMA50 slope < 0), volume confirmation
            elif close[i] < bb_lower[i] and ema_slope_4h_aligned[i] < 0 and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below BB lower OR trend reversal (EMA50 slope <= 0)
            if close[i] < bb_lower[i] or ema_slope_4h_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above BB upper OR trend reversal (EMA50 slope >= 0)
            if close[i] > bb_upper[i] or ema_slope_4h_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_BB20_2_4hEMA50Slope_1dATRVolConfirm_v1"
timeframe = "1h"
leverage = 1.0