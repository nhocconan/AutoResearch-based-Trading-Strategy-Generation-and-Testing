#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d ADX trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (strong trend) AND volume > 1.5x 20 EMA
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (strong trend) AND volume > 1.5x 20 EMA
# Uses 6h for entry timing, 1d for trend strength to avoid choppy markets.
# Discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Works in bull markets via longs in strong uptrends and bear markets via shorts in strong downtrends.

name = "6h_WilliamsR_Extreme_1dADX_VolumeConfirm"
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
    
    # Get 1d data for Williams %R and ADX calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    
    # Calculate ADX (14-period)
    # ADX requires +DI and -DI calculation
    # +DM = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
    # -DM = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
    # TR = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    # +DI = 100 * EMA(+DM) / ATR
    # -DI = 100 * EMA(-DM) / ATR
    # ADX = EMA(|+DI - -DI| / (+DI + -DI))
    
    # Calculate +DM and -DM
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr3 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR (14-period EMA of TR)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d Williams %R and ADX to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND ADX > 25 (strong trend) AND volume spike
            if (williams_r_aligned[i] < -80 and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND ADX > 25 (strong trend) AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -20 (overbought) OR ADX < 20 (weak trend)
            if (williams_r_aligned[i] > -20 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -80 (oversold) OR ADX < 20 (weak trend)
            if (williams_r_aligned[i] < -80 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals