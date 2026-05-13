#!/usr/bin/env python3
"""
6h_VWAP_Trend_Reversal_With_DMI
Hypothesis: Use 6h VWAP deviation combined with daily DMI (ADX) regime filter to capture mean-reversion in ranging markets and trend continuation in strong trends. Long when price is below VWAP and ADX < 20 (range) with bullish DI crossover; short when price above VWAP and ADX < 20 with bearish DI crossover. In trending markets (ADX >= 25), follow the trend: long when price > VWAP and +DI > -DI, short when price < VWAP and -DI > +DI. This adapts to both bull and bear markets by switching between mean-reversion and trend-following based on ADX regime.
"""

name = "6h_VWAP_Trend_Reversal_With_DMI"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for DMI (ADX) calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate DMI (ADX) components
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    period = 14
    if len(plus_dm) < period:
        return np.zeros(n)
    
    smoothed_plus_dm = wilders_smoothing(plus_dm, period)
    smoothed_minus_dm = wilders_smoothing(minus_dm, period)
    smoothed_tr = wilders_smoothing(tr, period)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Handle NaN and inf
    plus_di = np.nan_to_num(plus_di, nan=0.0)
    minus_di = np.nan_to_num(minus_di, nan=0.0)
    dx = np.nan_to_num(dx, nan=0.0)
    
    # ADX is smoothed DX
    adx = wilders_smoothing(dx, period)
    
    # Align DMI to 6h timeframe
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate VWAP for 6h data
    # Typical price
    typical_price = (high + low + close) / 3.0
    # VWAP = cumulative(typical_price * volume) / cumulative(volume)
    cum_vol = np.cumsum(volume)
    cum_vol_price = np.cumsum(typical_price * volume)
    vwap = np.where(cum_vol > 0, cum_vol_price / cum_vol, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        # Regime classification
        is_ranging = adx_aligned[i] < 20
        is_trending = adx_aligned[i] >= 25
        
        if position == 0:
            # Entry conditions
            if is_ranging:
                # Mean reversion in ranging market
                if close[i] < vwap[i] and plus_di_aligned[i] > minus_di_aligned[i]:
                    # Bullish bias: price below VWAP + bullish DI crossover
                    signals[i] = 0.25
                    position = 1
                elif close[i] > vwap[i] and minus_di_aligned[i] > plus_di_aligned[i]:
                    # Bearish bias: price above VWAP + bearish DI crossover
                    signals[i] = -0.25
                    position = -1
            elif is_trending:
                # Trend following in trending market
                if close[i] > vwap[i] and plus_di_aligned[i] > minus_di_aligned[i]:
                    # Uptrend: price above VWAP + bullish DI dominance
                    signals[i] = 0.25
                    position = 1
                elif close[i] < vwap[i] and minus_di_aligned[i] > plus_di_aligned[i]:
                    # Downtrend: price below VWAP + bearish DI dominance
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: VWAP cross or DI crossover
            if close[i] < vwap[i] or minus_di_aligned[i] > plus_di_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VWAP cross or DI crossover
            if close[i] > vwap[i] or plus_di_aligned[i] > minus_di_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals