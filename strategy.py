#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trend filter
# Uses Donchian channels from prior completed 4h bar for structure (breakout = momentum)
# Volume confirmation (>1.5x 20-period EMA volume) ensures breakout has participation
# ATR(14) trend filter: only long when price > EMA(50) + 0.5*ATR, short when price < EMA(50) - 0.5*ATR
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
# Focus on BTC/ETH by avoiding SOL-only bias through volume and trend requirements

name = "4h_Donchian20_VolumeConfirm_ATRTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (prior completed bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough data for Donchian calculation
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels from prior completed 4h bar
    # Upper = max(high) over last 20 periods, Lower = min(low) over last 20 periods
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed 4h bar (no look-ahead)
    upper_20_shifted = np.roll(upper_20, 1)
    lower_20_shifted = np.roll(lower_20, 1)
    upper_20_shifted[0] = np.nan
    lower_20_shifted[0] = np.nan
    
    # Align Donchian levels to 4h timeframe (same timeframe, so direct alignment)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20_shifted)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) for trend filter and volatility adjustment
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # EMA(50) for trend direction
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Dynamic trend bands: EMA(50) ± 0.5*ATR(14)
    upper_trend = ema_50 + 0.5 * atr_14
    lower_trend = ema_50 - 0.5 * atr_14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(ema_50[i]) or np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND volume spike AND price > upper trend band
            if close[i] > upper_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]) and close[i] > upper_trend[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND volume spike AND price < lower trend band
            elif close[i] < lower_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]) and close[i] < lower_trend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian lower OR price < EMA(50)
            if close[i] < lower_aligned[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian upper OR price > EMA(50)
            if close[i] > upper_aligned[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals