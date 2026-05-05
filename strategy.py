#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX Regime Filter
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (trending)
# Short when Bull Power < 0 AND Bear Power > 0 AND 12h ADX > 25 (trending)
# Exit when Elder Ray signals weaken OR 12h ADX < 20 (range regime)
# Uses 12h ADX to filter for trending markets only, reducing whipsaws in ranging markets
# Elder Ray provides clear bull/bear power measurement effective in both bull and bear markets
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 6h (primary), HTF: 12h

name = "6h_ElderRay_12hADX_Regime_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data ONCE before loop for ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values using Wilder's smoothing (similar to EMA but different factor)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period])  # Skip first NaN
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
                else:
                    result[i] = np.nan
            return result
        
        atr = wilders_smoothing(tr, period)
        dm_plus_smooth = wilders_smoothing(dm_plus, period)
        dm_minus_smooth = wilders_smoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 
                      100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = wilders_smoothing(dx, period)
        return adx
    
    # Calculate Elder Ray on 6h timeframe
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Calculate 12h ADX
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (strong trend)
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                adx_12h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 AND 12h ADX > 25 (strong trend)
            elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                  adx_12h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Elder Ray weakens OR 12h ADX < 20 (range regime)
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Elder Ray weakens OR 12h ADX < 20 (range regime)
            if (bull_power[i] >= 0 or bear_power[i] <= 0 or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals