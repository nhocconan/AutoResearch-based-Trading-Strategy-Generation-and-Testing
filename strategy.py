#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 and Bear Power < 0 (both positive momentum) AND 1d ADX > 25 (trending)
# Short when Bull Power < 0 and Bear Power > 0 (both negative momentum) AND 1d ADX > 25 (trending)
# Uses 1d ADX to filter for trending markets only, avoiding whipsaws in ranging conditions.
# Elder Ray captures the strength of bulls/bears via price relative to EMA.
# Discrete position sizing (0.25) to manage drawdown and minimize fee churn.
# Target: 50-150 total trades over 4 years on 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: ADX(14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # first value is simple average
        result[period-1] = np.nanmean(data[:period])
        # subsequent values: EMA-like with alpha=1/period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 13) + 5  # ADX needs ~34 bars (14+14+6), EMA13 needs 13
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (high > EMA13) - bulls in control
        # 2. Bear Power < 0 (low < EMA13) - bears weak
        # 3. 1d ADX > 25 - strong trending market
        if (bull_power[i] > 0) and (bear_power[i] < 0) and (adx_aligned[i] > 25):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bull Power < 0 (high < EMA13) - bulls weak
        # 2. Bear Power > 0 (low > EMA13) - bears in control
        # 3. 1d ADX > 25 - strong trending market
        elif (bull_power[i] < 0) and (bear_power[i] > 0) and (adx_aligned[i] > 25):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_1dADX_Regime_Filter_v1"
timeframe = "6h"
leverage = 1.0