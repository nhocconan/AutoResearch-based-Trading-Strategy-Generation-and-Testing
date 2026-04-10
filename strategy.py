#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter with volume confirmation
# - Elder Ray Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power rising (less negative) AND 12h ADX > 25 AND volume > 1.5x 20-period average
# - Short when Bear Power < 0 AND Bull Power falling (less positive) AND 12h ADX > 25 AND volume > 1.5x 20-period average
# - Exit when Elder Power reverses sign or ADX < 20 (regime change)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear power relative to EMA, effective in both trending and ranging markets
# - ADX filter ensures we only trade in strong trends, reducing whipsaws
# - Volume confirmation adds conviction to moves

name = "6h_12h_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = ema_13 - low   # Bear Power = EMA - Low
    
    # Pre-compute 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h ADX(14) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range and Directional Movement
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/14)
    def wilders_smoothing(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.mean(arr[1:period])  # First value
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    plus_di = 100 * dm_plus_14 / tr_14
    minus_di = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = wilders_smoothing(dx, 14)
    
    # ADX regime: strong trend when ADX > 25
    strong_trend = adx > 25
    weak_trend = adx < 20  # For exit
    
    # Align HTF indicators to 6h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_12h, strong_trend)
    weak_trend_aligned = align_htf_to_ltf(prices, df_12h, weak_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(strong_trend_aligned[i]) or 
            np.isnan(weak_trend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bear Power rising (less negative) AND strong trend AND volume spike
            bear_power_rising = (i > 1 and bear_power[i] > bear_power[i-1])
            if (bull_power[i] > 0 and 
                bear_power_rising and 
                strong_trend_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power < 0 AND Bull Power falling (less positive) AND strong trend AND volume spike
            elif (bear_power[i] < 0 and 
                  (i > 1 and bull_power[i] < bull_power[i-1]) and 
                  strong_trend_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Elder Power reverses sign OR trend weakens
            exit_long = (position == 1 and (bull_power[i] <= 0 or weak_trend_aligned[i]))
            exit_short = (position == -1 and (bear_power[i] >= 0 or weak_trend_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals