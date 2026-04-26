#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with 1d regime filter (ADX<20 = range, ADX>25 = trend). In range markets, fade extremes (sell Bull Power > 0.8*ATR, buy Bear Power < -0.8*ATR). In trending markets, follow momentum (buy Bull Power > 0, sell Bear Power < 0). Uses 6h timeframe for balance of signal frequency and noise reduction. Volume confirmation ensures institutional participation. Fixed size 0.25 to control trade frequency (~15-25 trades/year). Works in both bull and bear regimes by adapting to market state.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (using 6h close)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ATR(10) for volatility normalization
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1d ADX(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr1_1d = np.abs(high_1d[1:] - low_1d[:-1])
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([tr1_1d[0], tr2_1d[0], tr3_1d[0]])], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14_1d = wilder_smooth(tr_1d, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus_1d = 100 * dm_plus_14 / (tr14_1d + 1e-10)
    di_minus_1d = 100 * dm_minus_14 / (tr14_1d + 1e-10)
    
    # DX and ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d + 1e-10)
    adx_1d = wilder_smooth(dx_1d, 14)
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 50th percentile of 20-period lookback
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_ok = volume > vol_median
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (20 for volume median, 13 for EMA, 10 for ATR, ~41 for ADX)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13[i]) or 
            np.isnan(at_r10[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_median[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        atr_val = atr10[i]
        adx_val = adx_1d_aligned[i]
        vol_ok = volume_ok[i]
        size = fixed_size
        
        # Regime detection: ADX < 20 = range, ADX > 25 = trend
        is_range = adx_val < 20
        is_trend = adx_val > 25
        
        # Initialize signal
        signals[i] = 0.0
        
        if position == 0:
            # Flat - look for entry
            if vol_ok:
                if is_range:
                    # In range: fade extremes
                    if bull_val > 0.8 * atr_val:  # Overbought - sell
                        signals[i] = -size
                        position = -1
                    elif bear_val > 0.8 * atr_val:  # Oversold - buy
                        signals[i] = size
                        position = 1
                elif is_trend:
                    # In trend: follow momentum
                    if bull_val > 0:  # Bullish momentum
                        signals[i] = size
                        position = 1
                    elif bear_val > 0:  # Bearish momentum
                        signals[i] = -size
                        position = -1
        elif position == 1:
            # Long - exit conditions
            if is_range:
                # In range: exit when Bull Power normalizes
                if bull_val < 0.2 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            elif is_trend:
                # In trend: exit when momentum fades
                if bull_val <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Short - exit conditions
            if is_range:
                # In range: exit when Bear Power normalizes
                if bear_val < 0.2 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            elif is_trend:
                # In trend: exit when momentum fades
                if bear_val <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime"
timeframe = "6h"
leverage = 1.0