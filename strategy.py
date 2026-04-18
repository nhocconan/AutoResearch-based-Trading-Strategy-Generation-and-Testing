#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_VolumeTrend
1d strategy using Keltner Channel breakouts with volume confirmation and weekly trend filter.
- Long: Close breaks above upper KC(20,2) + volume > 1.5x weekly avg + weekly EMA50 > EMA200
- Short: Close breaks below lower KC(20,2) + volume > 1.5x weekly avg + weekly EMA50 < EMA200
- Exit: Opposite breakout or trend reversal
Designed for ~10-25 trades/year per symbol (40-100 total over 4 years)
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
    
    # Get weekly data for trend filter and volume average
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA50 and EMA200 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    ema_50_w = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_w = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Weekly volume average (20-period)
    vol_ma_20w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_w = align_htf_to_ltf(prices, df_1w, vol_ma_20w)
    
    # Daily Keltner Channel (20,2)
    atr_period = 20
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    ma_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ma_20 + 2 * atr
    kc_lower = ma_20 - 2 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # need enough for MA20 and weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_50_w[i]) or np.isnan(ema_200_w[i]) or
            np.isnan(vol_ma_w[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_w[i] > ema_200_w[i]
        downtrend = ema_50_w[i] < ema_200_w[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_w[i]
        
        # Breakout conditions
        breakout_up = close[i] > kc_upper[i]
        breakdown_down = close[i] < kc_lower[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above upper KC
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below lower KC
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or breakdown below lower KC
            if not uptrend or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or breakout above upper KC
            if not downtrend or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Keltner_Channel_Breakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0