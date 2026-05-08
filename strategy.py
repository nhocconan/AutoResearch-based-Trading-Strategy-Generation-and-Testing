#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA with 1d volume and ADX filter. Uses KAMA to capture trend direction while filtering out choppy markets.
# Long when KAMA is rising (bullish trend) AND 1d volume > 1.5x 20-day average AND ADX(14) > 20 (trending market).
# Short when KAMA is falling (bearish trend) AND 1d volume > 1.5x 20-day average AND ADX(14) > 20.
# Exit when KAMA changes direction or filters fail.
# KAMA adapts to market noise, reducing false signals in ranging markets while capturing trends.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "12h_KAMA_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Daily data for volume and ADX
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) on 12h data
    # Parameters: ER period=10, Fast EMA=2, Slow EMA=30
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder for correct calc
    # Recalculate volatility properly
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Actually, we need to compute rolling sum of absolute changes
    # Let's compute ER properly using pandas for simplicity within constraints
    close_s = pd.Series(close)
    change = close_s.diff().abs()
    volatility_sum = change.rolling(window=er_period).sum()
    price_change = np.abs(close_s - close_s.shift(er_period))
    er = price_change / volatility_sum
    er = er.fillna(0).values
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: rising or falling
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # Daily volume filter: current volume > 1.5x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.5 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    # Daily ADX(14) for trend strength
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_d[0] - low_d[0]  # First TR
    
    # Directional Movement
    plus_dm = np.where((high_d - np.roll(high_d, 1)) > (np.roll(low_d, 1) - low_d), 
                       np.maximum(high_d - np.roll(high_d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_d, 1) - low_d) > (high_d - np.roll(high_d, 1)), 
                        np.maximum(np.roll(low_d, 1) - low_d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    tr_s = pd.Series(tr)
    atr = tr_s.rolling(window=14, min_periods=14).mean().values
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    plus_dm_smooth = plus_dm_s.rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = minus_dm_s.rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    # Handle division by zero
    plus_di = np.where(atr != 0, plus_di, 0)
    minus_di = np.where(atr != 0, minus_di, 0)
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) != 0, dx, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(adx)] = 0
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_d, adx)
    
    # Trend filter: ADX > 20
    trend_filter = adx_aligned > 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA rising, volume filter, trending market
            long_cond = kama_rising[i] and volume_filter[i] and trend_filter[i]
            # Short conditions: KAMA falling, volume filter, trending market
            short_cond = kama_falling[i] and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA falling or filters fail
            if not (kama_rising[i] and volume_filter[i] and trend_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA rising or filters fail
            if not (kama_falling[i] and volume_filter[i] and trend_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals