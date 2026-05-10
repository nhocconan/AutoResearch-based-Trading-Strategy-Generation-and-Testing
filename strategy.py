#!/usr/bin/env python3
"""
6h_ParabolicSAR_Trend_With_ADX_Filter
Hypothesis: Uses Parabolic SAR to identify 6h trend direction, filtered by ADX for
trend strength. Only takes trades when ADX > 25 (strong trend) and SAR confirms
direction. In bull markets, rides uptrends; in bear markets, captures short
selling opportunities during downtrends. Uses 1d trend as higher timeframe filter
to avoid counter-trend trades. Target: 20-40 trades/year per symbol.
"""

name = "6h_ParabolicSAR_Trend_With_ADX_Filter"
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
    
    # Convert to Series for indicator calculations
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Parabolic SAR
    # Start with acceleration factor
    af = 0.02
    max_af = 0.2
    # Initialize
    sar = np.zeros(n)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    ep = np.zeros(n)     # extreme point
    
    # Set initial values
    sar[0] = low[0]
    trend[0] = 1
    ep[0] = high[0]
    
    for i in range(1, n):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
            # SAR cannot exceed previous two lows
            sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
            
            if low[i] < sar[i]:  # trend reversal
                trend[i] = -1
                sar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = low[i]
                af = 0.02
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af = min(af + 0.02, max_af)
                else:
                    ep[i] = ep[i-1]
                    af = af
        else:  # downtrend
            sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
            # SAR cannot go below previous two highs
            sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
            
            if high[i] > sar[i]:  # trend reversal
                trend[i] = 1
                sar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = high[i]
                af = 0.02
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af = min(af + 0.02, max_af)
                else:
                    ep[i] = ep[i-1]
                    af = af
    
    # ADX calculation
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i] / period
        return result
    
    period = 14
    atr = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(atr != 0, atr, 1e-10)
    di_minus = 100 * dm_minus_smooth / np.where(atr != 0, atr, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), 1e-10)
    adx = wilder_smooth(dx, period)
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sar[i]) or np.isnan(trend[i]) or np.isnan(adx[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Enter long: SAR bullish + strong trend + 1d uptrend
            if (trend[i] == 1 and strong_trend and trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Enter short: SAR bearish + strong trend + 1d downtrend
            elif (trend[i] == -1 and strong_trend and trend_1d_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when trend reverses or ADX weakens
            if (trend[i] == -1 or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when trend reverses or ADX weakens
            if (trend[i] == 1 or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals