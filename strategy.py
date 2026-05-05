#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d ADX Trend Filter and Volume Spike
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 2.0x 20 EMA
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 2.0x 20 EMA
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year per symbol.
# Works in bull markets via longs on pullbacks and bear markets via shorts on rallies.
# Uses 1d for HTF trend to avoid counter-trend trades and 6h for entry timing.

name = "6h_WilliamsR_Extreme_1dADX_VolumeSpike"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Williams %R and ADX
        return np.zeros(n)
    
    # Get daily OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    
    # Williams %R extremes: < -80 oversold, > -20 overbought
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    
    # Calculate 1d ADX (14-period) for trend strength
    # ADX requires +DI and -DI calculation
    # +DI = 100 * EWM of (+DM / TR)
    # -DI = 100 * EWM of (-DM / TR)
    # DX = 100 * |(+DI - -DI)| / ((+DI) + (-DI))
    # ADX = EWM of DX
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EWM with alpha=1/period)
    def wilder_smooth(data, period):
        """Wilder's smoothing: similar to EWM with alpha=1/period"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: prev * (period-1)/period + current/period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di_14 = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di_14 = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 0)
    adx = wilder_smooth(dx, 14)
    
    # Strong trend when ADX > 25
    strong_trend = adx > 25
    
    # Align 1d indicators to 6h timeframe
    williams_oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_oversold.astype(float))
    williams_overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_overbought.astype(float))
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_oversold_aligned[i]) or np.isnan(williams_overbought_aligned[i]) or 
            np.isnan(strong_trend_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold AND strong trend AND volume spike
            if (williams_oversold_aligned[i] > 0.5 and 
                strong_trend_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought AND strong trend AND volume spike
            elif (williams_overbought_aligned[i] > 0.5 and 
                  strong_trend_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 OR trend weakens
            if (williams_oversold_aligned[i] < 0.5 or  # Not oversold anymore
                strong_trend_aligned[i] < 0.5):        # Trend weakened
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 OR trend weakens
            if (williams_overbought_aligned[i] < 0.5 or  # Not overbought anymore
                strong_trend_aligned[i] < 0.5):          # Trend weakened
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals