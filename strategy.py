#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter. Long when Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (trending). Short when Bear Power < 0 AND Bull Power > 0 AND 12h ADX > 25. Uses 6h EMA13 for power calculation. Position size 0.25. Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions. Elder Ray measures bull/bear strength relative to EMA, ADX filters for trending markets only to avoid whipsaws in ranging markets. Works in both bull and bear markets by only trading in direction of Elder Ray when ADX confirms trend strength.

name = "6h_ElderRay_ADX_Regime"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h ADX with period 14
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(values[period-1:2*period-1]) if not np.all(np.isnan(values[period-1:2*period-1])) else np.nan
            # Rest is Wilder smoothing
            alpha = 1.0 / period
            for i in range(period, len(values)):
                if np.isnan(result[i-1]):
                    result[i] = np.nan
                elif np.isnan(values[i]):
                    result[i] = result[i-1] * (1 - alpha)
                else:
                    result[i] = values[i] * alpha + result[i-1] * (1 - alpha)
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h != 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h != 0, 100 * dm_minus_smooth / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_12h = wilders_smoothing(dx, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema_13_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # ADX trend filter: only trade when ADX > 25 (strong trend)
        trending = adx_12h_aligned[i] > 25
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND trending
            if bull_power[i] > 0 and bear_power[i] < 0 and trending:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND trending
            elif bear_power[i] < 0 and bull_power[i] > 0 and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Elder Ray turns bearish OR ADX weakens
            if bull_power[i] <= 0 or bear_power[i] >= 0 or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Elder Ray turns bullish OR ADX weakens
            if bear_power[i] >= 0 or bull_power[i] <= 0 or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals