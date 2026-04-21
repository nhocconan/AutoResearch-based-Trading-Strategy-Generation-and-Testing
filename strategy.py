#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_ATRFilter_V1
Hypothesis: Donchian(20) breakout with volume confirmation and ATR-based trend filter on 4h timeframe.
Works in bull/bear markets: breakouts capture strong moves, volume filter avoids false breakouts,
ATR filter ensures trading in the direction of volatility expansion. Uses 1d EMA50 for trend bias.
Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend bias
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility filter and stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr = np.zeros_like(tr)
    atr[:] = np.nan
    if len(tr) >= 14:
        # Initial ATR as simple average of first 14 TR values
        atr[13] = np.nanmean(tr[1:15])
        # Wilder's smoothing: ATR[t] = (ATR[t-1] * 13 + TR[t]) / 14
        for i in range(14, len(tr)):
            if not np.isnan(tr[i]) and not np.isnan(atr[i-1]):
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback-1, len(high)):
        if not np.isnan(high[i-lookback+1:i+1]).any() and not np.isnan(low[i-lookback+1:i+1]).any():
            highest_high[i] = np.max(high[i-lookback+1:i+1])
            lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: 1d EMA50 slope
        if i >= 51:
            ema_rising = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            ema_falling = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
        else:
            ema_rising = True
            ema_falling = True
        
        if position == 0:
            # Long entry: price breaks above Donchian upper band + volume + 1d uptrend
            if (price > highest_high[i] and volume_ok and ema_rising):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower band + volume + 1d downtrend
            elif (price < lowest_low[i] and volume_ok and ema_falling):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower band or ATR-based stop
            # ATR stop: 2.5 * ATR below entry (tracked via position management)
            # For simplicity, exit when price retracement of 50% of the breakout move
            if i >= 1:
                breakout_level = highest_high[i-1]  # Previous bar's high was the breakout level
                retracement = price < (breakout_level - 0.5 * atr[i])
                if retracement or price < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper band or ATR-based stop
            if i >= 1:
                breakout_level = lowest_low[i-1]  # Previous bar's low was the breakdown level
                retracement = price > (breakout_level + 0.5 * atr[i])
                if retracement or price > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ATRFilter_V1"
timeframe = "4h"
leverage = 1.0