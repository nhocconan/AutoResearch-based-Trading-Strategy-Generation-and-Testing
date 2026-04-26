#!/usr/bin/env python3
"""
6h_ADX_Trend_ElderRay_Reversal
Hypothesis: On 6h timeframe, use 1d ADX > 25 to identify strong trends, then fade Elder Ray Bull/Bear power extremes for mean reversion entries.
In strong trends (ADX>25), Elder Ray power often overextends before continuation - we fade these extremes.
Long when Bear Power crosses above -ATR(10) (less negative) with ADX>25 and price>EMA20(1d).
Short when Bull Power crosses below ATR(10) (less positive) with ADX>25 and price<EMA20(1d).
Exit when Elder Ray power returns toward zero (mean reversion complete) or ADX weakens (<20).
Uses discrete sizing (0.25) to minimize fees. Target: 80-180 trades over 4 years.
Works in bull/bear by trading mean reversion within strong trends, avoiding choppy markets via ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for HTF indicators (ADX, EMA20, Elder Ray components)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA20 for trend bias
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 1d ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                           np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+ , DM- using Wilder's smoothing (EMA with alpha=1/period)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
                    else:
                        result[i] = result[i-1]
            return result
        
        atr = WilderSmoothing(tr, period)
        dm_plus_smooth = WilderSmoothing(dm_plus, period)
        dm_minus_smooth = WilderSmoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = WilderSmoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d ATR(10) for Elder Ray scaling
    def calculate_atr(high, low, close, period=10):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
                    else:
                        result[i] = result[i-1]
            return result
        
        atr = WilderSmoothing(tr, period)
        return atr
    
    atr10_1d = calculate_atr(high_1d, low_1d, close_1d, 10)
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # 1d Elder Ray Components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period EMA, 14-period ADX, 10-period ATR, 13-period EMA for Elder Ray)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(atr10_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long entry: Bear Power crosses above -ATR(10) (less negative) in strong uptrend
        # Conditions: ADX>25 (strong trend), price>EMA20 (uptrend bias), Bear Power > -ATR(10) and rising
        if (adx_1d_aligned[i] > 25 and 
            close[i] > ema20_1d_aligned[i] and
            bear_power_aligned[i] > -atr10_1d_aligned[i] and
            i > start_idx and bear_power_aligned[i-1] <= -atr10_1d_aligned[i-1]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        
        # Short entry: Bull Power crosses below ATR(10) (less positive) in strong downtrend
        # Conditions: ADX>25 (strong trend), price<EMA20 (downtrend bias), Bull Power < ATR(10) and falling
        elif (adx_1d_aligned[i] > 25 and 
              close[i] < ema20_1d_aligned[i] and
              bull_power_aligned[i] < atr10_1d_aligned[i] and
              i > start_idx and bull_power_aligned[i-1] >= atr10_1d_aligned[i-1]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        
        # Exit conditions: mean reversion complete or trend weakening
        elif position == 1:
            # Exit long: Bear Power returns toward zero (oversold condition unwinding) OR ADX weakens
            if (bear_power_aligned[i] >= 0 or adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Exit short: Bull Power returns toward zero (overbought condition unwinding) OR ADX weakens
            if (bull_power_aligned[i] <= 0 or adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
        else:
            # Flat
            signals[i] = 0.0
    
    return signals

name = "6h_ADX_Trend_ElderRay_Reversal"
timeframe = "6h"
leverage = 1.0