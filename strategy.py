#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Long when price breaks above Camarilla H3 AND close > 4h EMA50 (bullish trend)
- Short when price breaks below Camarilla L3 AND close < 4h EMA50 (bearish trend)
- Volume must be > 1.3x ATR(14) * close (volatility-adjusted volume filter)
- Exit on trend reversal or price retracing to Camarilla Pivot point
- Uses 1h primary timeframe with 4h HTF for trend direction, targeting 60-150 trades over 4 years (15-37/year)
- Camarilla levels provide intraday support/resistance that work in ranging markets
- 4h EMA50 filters for intermediate trend alignment to reduce whipsaws
- ATR-scaled volume confirmation adapts to volatility, reducing false breakouts
- Session filter (08-20 UTC) avoids low-liquidity Asian session noise
- Designed for BTC/ETH with edge in ranging markets (mean reversion at H3/L3) and trending markets (breakout continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels using previous day's OHLC (no look-ahead)
    # Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # Use daily OHLC from previous completed day
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Get daily OHLC (using 1d timeframe for Camarilla calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align daily Camarilla levels to 1h timeframe
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3  # Standard pivot
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(14) for dynamic volume threshold
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 1.3 * ATR * close (volatility-adjusted)
    vol_threshold = 1.3 * atr * close
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 14) + 1  # Camarilla needs 1d data, EMA50 needs 50, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
                                 np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
                                 np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3, trend up (close > EMA50), volume confirmation
            if close[i] > camarilla_h3_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla L3, trend down (close < EMA50), volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price retreats to Camarilla Pivot OR trend reverses
            if close[i] <= camarilla_pivot_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rallies to Camarilla Pivot OR trend reverses
            if close[i] >= camarilla_pivot_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_VolumeConfirm_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0