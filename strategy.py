#!/usr/bin/env python3
"""
6h_WeeklyDonchian_Breakout_1dTrendFilter_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout aligned with 1d EMA50 trend filter and volume confirmation (>1.5x 20-period average).
Long when price breaks above 20-period 6h high in 1-day uptrend with volume confirmation.
Short when price breaks below 20-period 6h low in 1-day downtrend with volume confirmation.
Exit via opposite Donchian boundary or ATR trailing stop (2.5*ATR from extreme).
Uses weekly trend as higher timeframe filter to avoid counter-trend trades in bear markets.
Designed for ~60-120 trades over 4 years (15-30/year) via tight breakout conditions with multi-timeframe alignment.
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
    
    # Get 1d data for trend filter and weekly data for higher trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly EMA20 for higher timeframe trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 6h Donchian channels (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, donchian_period, atr_period, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend_1d = ema_50_1d_aligned[i]
        ema_trend_1w = ema_20_1w_aligned[i]
        upper_donchian = highest_high[i]
        lower_donchian = lowest_low[i]
        
        if position == 0:
            # Only trade when both timeframes agree on trend direction
            if close[i] > ema_trend_1d and close[i] > ema_trend_1w:  # Bullish alignment
                # Long: break above Donchian high with volume confirmation
                long_signal = (close[i] > upper_donchian) and vol_regime[i]
            elif close[i] < ema_trend_1d and close[i] < ema_trend_1w:  # Bearish alignment
                # Short: break below Donchian low with volume confirmation
                short_signal = (close[i] < lower_donchian) and vol_regime[i]
            else:
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = long_extreme - 2.5 * atr[i]
            # 2. Price breaks below Donchian low (opposite boundary)
            if close[i] <= atr_stop or close[i] < lower_donchian:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = short_extreme + 2.5 * atr[i]
            # 2. Price breaks above Donchian high (opposite boundary)
            if close[i] >= atr_stop or close[i] > upper_donchian:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchian_Breakout_1dTrendFilter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0