#!/usr/bin/env python3
"""
1h_Donchian20_VolumeBreakout_TrendFilter_1h
1h strategy using Donchian channel breakouts with volume confirmation and 4h/1d trend filters.
- Long: Price breaks above Donchian(20) high + volume > 1.5x avg + 4h EMA34 > EMA89 + 1d close > SMA50
- Short: Price breaks below Donchian(20) low + volume > 1.5x avg + 4h EMA34 < EMA89 + 1d close < SMA50
- Exit: Opposite breakout or trend reversal on 4h
- Session filter: 08:00-20:00 UTC only
Designed for 15-30 trades/year per symbol (60-120 total over 4 years)
Uses 4h/1d for trend direction, 1h for precise entry timing
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA34 and EMA89 for trend
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_4h = pd.Series(close_4h).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    ema_89_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_89_4h)
    
    # Get 1d data for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # 1h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1h volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 89  # need enough for EMA89
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_89_4h_aligned[i]) or 
            np.isnan(sma_50_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend_4h = ema_34_4h_aligned[i] > ema_89_4h_aligned[i]
        downtrend_4h = ema_34_4h_aligned[i] < ema_89_4h_aligned[i]
        uptrend_1d = close[i] > sma_50_1d_aligned[i]
        downtrend_1d = close[i] < sma_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i]
        breakdown_down = close[i] < low_20[i]
        
        if position == 0:
            # Long: 4h/1d uptrend + volume + breakout above Donchian high
            if uptrend_4h and uptrend_1d and vol_confirm and breakout_up:
                signals[i] = 0.20
                position = 1
            # Short: 4h/1d downtrend + volume + breakdown below Donchian low
            elif downtrend_4h and downtrend_1d and vol_confirm and breakdown_down:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal on 4h or Donchian breakdown
            if not uptrend_4h or breakdown_down:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend reversal on 4h or Donchian breakout
            if not downtrend_4h or breakout_up:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_VolumeBreakout_TrendFilter_1h"
timeframe = "1h"
leverage = 1.0