# %%
#!/usr/bin/env python3
"""
6h_ADX_ElderRay_1DTrend_Filter
Hypothesis: On 6h timeframe, use daily trend direction (via EMA34) to filter Elder Ray (bull/bear power) signals, with ADX > 25 to ensure trending conditions. This avoids false signals in range and captures momentum in both bull and bear markets. The daily trend filter ensures we only trade in the direction of the higher timeframe trend, improving win rate. Target: 20-40 trades/year.
"""

name = "6h_ADX_ElderRay_1DTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 6h data for Elder Ray and ADX
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray (6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # ADX calculation (14 period)
    # +DM, -DM, TR
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    
    plus_dm = np.where((high - high_shift) > (low_shift - low), np.maximum(high - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low) > (high - high_shift), np.maximum(low_shift - low, 0), 0)
    tr = np.maximum(high - low, np.maximum(np.abs(high - high_shift), np.abs(low - low_shift)))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    if len(tr) < period_adx:
        return np.zeros(n)
    
    atr = WilderSmooth(tr, period_adx)
    plus_di = 100 * WilderSmooth(plus_dm, period_adx) / atr
    minus_di = 100 * WilderSmooth(minus_dm, period_adx) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmooth(dx, period_adx)
    
    # Align daily trend to 6h
    uptrend_1d = close > ema34_1d_aligned
    downtrend_1d = close < ema34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA13 (13), ADX (14*2=28 for smoothing), and daily EMA34
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Conditions
        strong_trend = adx[i] > 25
        bullish_momentum = bull_power[i] > 0
        bearish_momentum = bear_power[i] < 0
        
        if position == 0:
            # Long: daily uptrend, ADX strong, bullish momentum
            if uptrend_1d[i] and strong_trend and bullish_momentum:
                signals[i] = 0.25
                position = 1
            # Short: daily downtrend, ADX strong, bearish momentum
            elif downtrend_1d[i] and strong_trend and bearish_momentum:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend weakens or momentum fades
            if not (uptrend_1d[i] and strong_trend and bullish_momentum):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakens or momentum fades
            if not (downtrend_1d[i] and strong_trend and bearish_momentum):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
# %%