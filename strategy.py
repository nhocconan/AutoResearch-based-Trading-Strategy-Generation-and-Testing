#!/usr/bin/env python3
# 6h_pivot_reversion_v1
# Hypothesis: Mean reversion at daily pivot levels (R1/S1) with weekly trend filter.
# In ranging markets (weekly ADX < 25), price reverts to weekly pivot (PP).
# In trending markets (weekly ADX >= 25), price pulls back to daily R1/S1 before continuing.
# Weekly trend determined by ADX(14). Weekly pivot from prior week's OHLC.
# Entry: 6h close crosses daily R1 (for short) or S1 (for long) with volume > 1.3x average.
# Exit: Opposite pivot level touched or weekly trend changes.
# Target: 15-25 trades/year per symbol. Low frequency avoids fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_pivot_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily pivot levels (from prior day)
    # PP = (H + L + C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    daily_pp = np.full(n, np.nan)
    daily_r1 = np.full(n, np.nan)
    daily_s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        if not (np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            pp = (high[i-1] + low[i-1] + close[i-1]) / 3.0
            r1 = 2 * pp - low[i-1]
            s1 = 2 * pp - high[i-1]
            daily_pp[i] = pp
            daily_r1[i] = r1
            daily_s1[i] = s1
    
    # Volume filter: 1.3x 24-period average (4 days of 6h data)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.3 * vol_ma[i]
    
    # Get weekly data for ADX trend filter and pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly ADX(14) for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def _wilders_smoothing(x, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        result = np.full_like(x, np.nan)
        if len(x) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(x[1:period])
        # Rest is EMA
        for i in range(period, len(x)):
            if not np.isnan(x[i]):
                result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr_period = 14
    atr = _wilders_smoothing(tr, atr_period)
    
    dx = np.full_like(close_1w, np.nan)
    for i in range(atr_period, len(close_1w)):
        if not np.isnan(at[i]) and atr[i] > 0:
            di_plus = _wilders_smoothing(dm_plus, atr_period)[i]
            di_minus = _wilders_smoothing(dm_minus, atr_period)[i]
            dx[i] = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    
    adx = _wilders_smoothing(dx, atr_period)
    
    # Weekly pivot levels (from prior week)
    weekly_pp = np.full(len(close_1w), np.nan)
    weekly_r1 = np.full(len(close_1w), np.nan)
    weekly_s1 = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        if not (np.isnan(high_1w[i-1]) or np.isnan(low_1w[i-1]) or np.isnan(close_1w[i-1])):
            pp = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
            r1 = 2 * pp - low_1w[i-1]
            s1 = 2 * pp - high_1w[i-1]
            weekly_pp[i] = pp
            weekly_r1[i] = r1
            weekly_s1[i] = s1
    
    # Align weekly indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(24, 30) + 1  # vol_ma + weekly lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_pp[i]) or np.isnan(daily_r1[i]) or np.isnan(daily_s1[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(weekly_pp_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price touches weekly R1 or weekly trend turns weak (ADX < 20)
            if (close[i] >= weekly_r1_aligned[i] or adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price touches weekly S1 or weekly trend turns weak (ADX < 20)
            if (close[i] <= weekly_s1_aligned[i] or adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Determine market regime: trending (ADX >= 25) or ranging (ADX < 25)
            is_trending = adx_aligned[i] >= 25
            
            if is_trending:
                # In trending market: look for pullbacks to daily S1/R1
                # Long: pullback to daily S1 with bullish bias (close > weekly PP)
                if (close[i] <= daily_s1[i] and 
                    close[i] > daily_pp[i] and  # above daily PP shows resilience
                    vol_surge[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: pullback to daily R1 with bearish bias (close < weekly PP)
                elif (close[i] >= daily_r1[i] and 
                      close[i] < daily_pp[i] and  # below daily PP shows weakness
                      vol_surge[i]):
                    position = -1
                    signals[i] = -0.25
            else:
                # In ranging market: mean reversion to weekly PP
                # Long: price near weekly S1 with bullish reversal
                if (close[i] <= weekly_s1_aligned[i] * 1.001 and  # slight buffer
                    close[i] > weekly_s1_aligned[i] * 0.999 and
                    vol_surge[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price near weekly R1 with bearish reversal
                elif (close[i] >= weekly_r1_aligned[i] * 0.999 and 
                      close[i] < weekly_r1_aligned[i] * 1.001 and
                      vol_surge[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals