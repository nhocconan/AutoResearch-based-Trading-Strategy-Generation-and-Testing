#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index + 1d ADX regime filter + volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 AND volume > 1.5x 20-period average
# - Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 AND volume > 1.5x 20-period average
# - Exit when Elder Ray signals reverse (Bull Power < 0 for long, Bear Power < 0 for short)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear power relative to EMA, effective in both trending and ranging markets
# - 1d ADX filter ensures we trade only when there is significant trend strength on higher timeframe
# - Volume confirmation reduces false signals

name = "6h_1d_elder_ray_adx_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Pre-compute 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha=1/14)
    def wilders_smoothing(arr, period):
        result = np.zeros_like(arr)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[1:period])  # First value
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = np.zeros_like(dx)
    adx[13] = np.mean(dx[14:28]) if len(dx) >= 28 else np.mean(dx[14:]) if len(dx) > 14 else 0
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # ADX regime: strong trend when ADX > 25
    strong_trend_regime = adx > 25
    
    # Align HTF indicators to 6h timeframe
    strong_trend_regime_aligned = align_htf_to_ltf(prices, df_1d, strong_trend_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(strong_trend_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND strong trend regime AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                strong_trend_regime_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power > 0 AND Bull Power < 0 AND strong trend regime AND volume spike
            elif (bear_power[i] > 0 and 
                  bull_power[i] < 0 and 
                  strong_trend_regime_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Elder Ray signals reverse
            exit_long = (position == 1 and bull_power[i] < 0)
            exit_short = (position == -1 and bear_power[i] < 0)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals