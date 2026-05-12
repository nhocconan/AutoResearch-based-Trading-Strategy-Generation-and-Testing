#!/usr/bin/env python3
name = "6h_ParabolicSAR_ADX_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d EMA34 for trend ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Parabolic SAR calculation ===
    # Parameters: step=0.02, max=0.2
    psar = np.zeros(n)
    psar[0] = low[0]  # start with low
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    ep = high[0] if trend == 1 else low[0]  # extreme point
    
    for i in range(1, n):
        if trend == 1:  # uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR stays below recent lows
            psar[i] = min(psar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
            if low[i] < psar[i]:  # trend reversal
                trend = -1
                psar[i] = ep
                af = 0.02
                ep = low[i]
        else:  # downtrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR stays above recent highs
            psar[i] = max(psar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
            if high[i] > psar[i]:  # trend reversal
                trend = 1
                psar[i] = ep
                af = 0.02
                ep = high[i]
        
        # Update acceleration factor and extreme point
        if trend == 1:
            if high[i] > ep:
                ep = high[i]
                af = min(af + 0.02, 0.2)
        else:
            if low[i] < ep:
                ep = low[i]
                af = min(af + 0.02, 0.2)
    
    # === ADX calculation (14-period) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.zeros_like(arr)
        result[:period-1] = np.nan
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = smooth_wilder(tr, 14)
    plus_dm14 = smooth_wilder(plus_dm, 14)
    minus_dm14 = smooth_wilder(minus_dm, 14)
    
    # Directional Indicators
    plus_di = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth_wilder(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(psar[i]) or
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: PSAR long (price > PSAR), ADX > 25, price above 1d EMA
            if (close[i] > psar[i] and 
                adx[i] > 25 and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: PSAR short (price < PSAR), ADX > 25, price below 1d EMA
            elif (close[i] < psar[i] and 
                  adx[i] > 25 and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < PSAR or ADX < 20 or price below EMA
            if (close[i] < psar[i] or 
                adx[i] < 20 or
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > PSAR or ADX < 20 or price above EMA
            if (close[i] > psar[i] or 
                adx[i] < 20 or
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals