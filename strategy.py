#!/usr/bin/env python3
"""
4h_Keltner_Breakout_VolumeTrend
4h strategy using Keltner Channel breakouts with volume confirmation and trend filter.
- Long: Close breaks above upper Keltner band + volume > 1.5x 20-period avg + EMA34 > EMA200
- Short: Close breaks below lower Keltner band + volume > 1.5x 20-period avg + EMA34 < EMA200
- Exit: Opposite breakout or trend reversal (EMA34 crosses EMA200)
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Keltner Channels
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 19 + tr[i]) / 20  # 20-period ATR
    
    # 20-period EMA for Keltner middle line
    ema20 = np.zeros(n)
    ema20[0] = close[0]
    for i in range(1, n):
        ema20[i] = (close[i] * 0.1) + (ema20[i-1] * 0.9)
    
    # Keltner Channels (20, 2.0)
    keltner_upper = ema20 + (2.0 * atr)
    keltner_lower = ema20 - (2.0 * atr)
    
    # Get daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA34 and EMA200 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_34_aligned[i] > ema_200_aligned[i]
        downtrend = ema_34_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > keltner_upper[i]
        breakdown_down = close[i] < keltner_lower[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above upper Keltner band
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below lower Keltner band
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or breakdown below lower band
            if not uptrend or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or breakout above upper band
            if not downtrend or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Keltner_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0