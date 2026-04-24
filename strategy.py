#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based stoploss.
- Uses 4h timeframe (primary) and 1d HTF for EMA50 trend alignment
- Donchian channels calculated from 20-period high/low on 4h data
- Breakout logic: long when price crosses above upper band with volume confirmation, short when crosses below lower band
- Trend filter: only long when price > 1d EMA50, only short when price < 1d EMA50
- Volume confirmation: current volume > 1.5 * 20-period volume MA to filter low-quality breakouts
- Stoploss: ATR(14) based trailing stop - exit long when price drops 2.5*ATR from peak, exit short when price rises 2.5*ATR from trough
- Discrete signal size: 0.25 to balance return and risk while minimizing fee churn
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture strong momentum moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR calculation for dynamic stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_stop_price = 0.0
    short_stop_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14, 20)  # Need 1d EMA50, Donchian(20), ATR(14), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above upper Donchian band AND uptrend AND volume confirmation
            if close[i] > highest_high[i] and close[i-1] <= highest_high[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                long_stop_price = highest_high[i] - 2.5 * atr[i]  # Initial stop
            # Short: price crosses below lower Donchian band AND downtrend AND volume confirmation
            elif close[i] < lowest_low[i] and close[i-1] >= lowest_low[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                short_stop_price = lowest_low[i] + 2.5 * atr[i]   # Initial stop
        elif position == 1:
            # Update trailing stop for long position
            long_stop_price = max(long_stop_price, highest_high[i] - 2.5 * atr[i])
            # Exit long: price hits trailing stop or reverse signal
            if close[i] <= long_stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop for short position
            short_stop_price = min(short_stop_price, lowest_low[i] + 2.5 * atr[i])
            # Exit short: price hits trailing stop or reverse signal
            if close[i] >= short_stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_ATRStop_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0