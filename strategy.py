#!/usr/bin/env python3
"""
6h Elder Ray Power + 1d ADX Regime Filter
Hypothesis: Elder Ray Bull/Bear Power measures buying/selling pressure relative to EMA13.
In strong trends (ADX>25 on 1d), we take trades in the direction of both Elder Ray power and trend.
In ranging markets (ADX<20), we fade extreme Elder Ray readings for mean reversion.
This adaptive approach works in both bull and bear markets by regime.
6h timeframe targets 12-37 trades/year (50-150 over 4 years).
"""

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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter (trending vs ranging)
    # ADX calculation: +DM, -DM, TR, then smoothed
    dm_plus = np.where((df_1d['high'].diff()) > (df_1d['low'].diff().abs()), 
                       np.maximum(df_1d['high'].diff(), 0), 0)
    dm_minus = np.where((df_1d['low'].diff().abs()) > (df_1d['high'].diff()), 
                        np.maximum(-df_1d['low'].diff(), 0), 0)
    tr = np.maximum(
        df_1d['high'] - df_1d['low'],
        np.maximum(
            np.abs(df_1d['high'] - df_1d['close'].shift(1)),
            np.abs(df_1d['low'] - df_1d['close'].shift(1))
        )
    )
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(tr) >= period:
        tr_smooth = wilders_smoothing(tr, period)
        dm_plus_smooth = wilders_smoothing(dm_plus, period)
        dm_minus_smooth = wilders_smoothing(dm_minus, period)
        
        # Avoid division by zero
        dm_plus_di = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
        dm_minus_di = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
        dx = 100 * np.abs(dm_plus_di - dm_minus_di) / (dm_plus_di + dm_minus_di + 1e-10)
        adx = wilders_smoothing(dx, period)
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13  # negative values indicate bearish pressure
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(13, 30)  # EMA13, ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        adx_val = adx_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            # Look for entry signals
            if is_trending:
                # In trending markets: trade with trend and Elder Ray power
                # Long: positive bull power AND price above EMA13 (uptrend confirmation)
                # Short: negative bear power AND price below EMA13 (downtrend confirmation)
                long_entry = (bull_power[i] > 0) and (close[i] > ema_13[i])
                short_entry = (bear_power[i] < 0) and (close[i] < ema_13[i])
                
                if long_entry:
                    signals[i] = 0.25
                    position = 1
                elif short_entry:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # In ranging markets: fade extreme Elder Ray readings
                # Long: extremely negative bear power (oversold) AND price below EMA13
                # Short: extremely positive bull power (overbought) AND price above EMA13
                # Use -1.5 * ATR as threshold for extreme readings
                atr_6h = pd.Series(
                    np.maximum(
                        high - low,
                        np.maximum(
                            np.abs(high - np.roll(close, 1)),
                            np.abs(low - np.roll(close, 1))
                        )
                    )
                ).rolling(window=14, min_periods=14).mean().values
                
                if i >= 14 and not np.isnan(atr_6h[i]):
                    extreme_bear = bear_power[i] < (-1.5 * atr_6h[i])
                    extreme_bull = bull_power[i] > (1.5 * atr_6h[i])
                    
                    long_entry = extreme_bear and (close[i] < ema_13[i])
                    short_entry = extreme_bull and (close[i] > ema_13[i])
                    
                    if long_entry:
                        signals[i] = 0.25
                        position = 1
                    elif short_entry:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX between 20-25): no trading
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Elder Ray turns negative OR price crosses below EMA13
            if (bull_power[i] <= 0) or (close[i] < ema_13[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Elder Ray turns positive OR price crosses above EMA13
            if (bear_power[i] >= 0) or (close[i] > ema_13[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_ADXRegime"
timeframe = "6h"
leverage = 1.0