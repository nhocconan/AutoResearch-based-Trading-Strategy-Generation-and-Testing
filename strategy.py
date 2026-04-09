#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures bull/bear strength
# 1d ADX > 25 = trending regime (follow Elder Ray signals), ADX < 20 = ranging regime (fade Elder Ray extremes)
# Works in bull/bear: regime filter adapts, Elder Ray captures momentum exhaustion and continuation
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray components
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(atr_1d != 0, 100 * plus_dm_smooth / atr_1d, 0)
    minus_di = np.where(atr_1d != 0, 100 * minus_dm_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar close)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13_6h
    bear_power = low - ema13_6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13_6h[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending (follow Elder Ray), ADX < 20 = ranging (fade Elder Ray)
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit: Bear power turns negative OR regime shifts to ranging
            if bear_power[i] < 0 or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull power turns negative OR regime shifts to ranging
            if bull_power[i] > 0 or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow Elder Ray in trending regime: strong bull/bear power
                if bull_power[i] > 0 and bear_power[i] < 0:  # Both confirm trend
                    if bull_power[i] > abs(bear_power[i]):  # Bullish bias
                        position = 1
                        signals[i] = 0.25
                    elif abs(bear_power[i]) > bull_power[i]:  # Bearish bias
                        position = -1
                        signals[i] = -0.25
            elif ranging_regime:
                # Fade Elder Ray extremes in ranging regime
                if bull_power[i] < 0 and bear_power[i] > 0:  # Both negative = exhaustion
                    if abs(bear_power[i]) > abs(bull_power[i]):  # More bearish exhaustion -> long
                        position = 1
                        signals[i] = 0.25
                    elif abs(bull_power[i]) > abs(bear_power[i]):  # More bullish exhaustion -> short
                        position = -1
                        signals[i] = -0.25
    
    return signals