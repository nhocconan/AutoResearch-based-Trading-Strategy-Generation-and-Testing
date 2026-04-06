#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + ADX Trend Filter
Hypothesis: Donchian breakouts capture momentum; volume spikes confirm institutional participation;
ADX > 25 ensures trending markets to avoid whipsaw in ranges. Designed for low trade frequency
(target 75-200 total over 4 years) to minimize fee decay. Works in bull (breakouts) and bear
(trend continuation) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 14-period ADX
    adx = np.full(n, np.nan)
    if n >= 14:
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr_14 = atr  # already calculated
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        if n >= 14:
            plus_sm = np.full(n, np.nan)
            minus_sm = np.full(n, np.nan)
            # Initial smoothed values
            plus_sm[13] = np.sum(plus_dm[1:14])
            minus_sm[13] = np.sum(minus_dm[1:14])
            for i in range(14, n):
                plus_sm[i] = plus_sm[i-1] - (plus_sm[i-1] / 14) + plus_dm[i]
                minus_sm[i] = minus_sm[i-1] - (minus_sm[i-1] / 14) + minus_dm[i]
            
            # Avoid division by zero
            plus_di[13:] = (plus_sm[13:] / atr_14[13:]) * 100
            minus_di[13:] = (minus_sm[13:] / atr_14[13:]) * 100
            
            # DX and ADX
            dx = np.full(n, np.nan)
            dx[13:] = np.abs(plus_di[13:] - minus_di[13:]) / (plus_di[13:] + minus_di[13:]) * 100
            # Wilder smoothing for ADX
            adx[27] = np.mean(dx[14:28])  # first ADX value
            for i in range(28, n):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 28  # For ADX
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # ADX trend filter: only trade when trending (ADX > 25)
        trend_filter = adx[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + trend filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if bull_breakout and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals