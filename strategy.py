#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend, 1d for volume spike filter.
- Donchian breakout captures momentum in both bull and bear markets.
- EMA50 on 12h filters breakouts to only trade in direction of higher timeframe trend.
- Volume confirmation (current volume > 1.5 * 20-period MA) avoids false breakouts.
- ATR-based stoploss: exit when price moves against position by 2.5 * ATR(20).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1d
    volume_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(20) for stoploss
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(low.shift())).abs()
    tr3 = (pd.Series(low) - pd.Series(close.shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current 4h volume > 1.5 * 20-period 1d volume MA
    volume_spike = volume > (1.5 * volume_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 12h bars for EMA and 20 for Donchian/ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish breakout: price closes above upper Donchian AND above 12h EMA50
                if curr_close > highest_high[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price closes below lower Donchian AND below 12h EMA50
                elif curr_close < lowest_low[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long exit: price closes below lower Donchian OR ATR stoploss hit
            if curr_close < lowest_low[i] or curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper Donchian OR ATR stoploss hit
            if curr_close > highest_high[i] or curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0