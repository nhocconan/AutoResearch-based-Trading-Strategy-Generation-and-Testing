#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h EMA34 trend filter and volume confirmation.
- Long when price breaks above Camarilla R3 level AND close > 4h EMA34 (bullish trend)
- Short when price breaks below Camarilla S3 level AND close < 4h EMA34 (bearish trend)
- Volume must be > 2.0x 20-period average for confirmation (tight filter)
- Uses 1h primary timeframe with 4h HTF to target 60-150 trades over 4 years (15-37/year)
- Session filter: only trade 08-20 UTC to avoid low-volume Asian session noise
- Camarilla pivots provide institutional support/resistance levels
- EMA34 trend filter ensures alignment with higher timeframe momentum
- Volume confirmation reduces false breakouts
- Designed for BTC/ETH with edge in both bull (breakout continuation) and bear (mean reversion at extremes) regimes
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
    
    # Calculate Camarilla levels using previous period (no look-ahead)
    prev_close = pd.Series(close).shift(1).values
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels: Close ± (Range * multiplier)
    camarilla_r3 = prev_close + (prev_range * 1.1 / 4)  # R3 = C + (H-L)*1.1/4
    camarilla_s3 = prev_close - (prev_range * 1.1 / 4)  # S3 = C - (H-L)*1.1/4
    
    # Get 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA34 to 1h timeframe
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * vol_ma
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3, trend up (close > EMA34), volume confirmation
            if close[i] > camarilla_r3[i] and close[i] > ema_34_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3, trend down (close < EMA34), volume confirmation
            elif close[i] < camarilla_s3[i] and close[i] < ema_34_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla S3 (mean reversion) OR trend reverses
            if close[i] < camarilla_s3[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above Camarilla R3 (mean reversion) OR trend reverses
            if close[i] > camarilla_r3[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_4hEMA34_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0