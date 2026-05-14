#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and 1d volume spike confirmation.
# Long when price breaks above upper Donchian channel AND 1d ATR(14)/ATR(50) > 0.8 (low volatility regime) AND 1d volume > 1.5 * 20-period average volume.
# Short when price breaks below lower Donchian channel AND same filters.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 50-150 total trades over 4 years (12-37/year) for 12h.
# Works in both bull and bear markets: ATR regime filter ensures we only trade in low-volatility environments where breakouts are more likely to sustain,
# while volume confirmation avoids false breakouts in low-participation environments.

name = "12h_Donchian20_Breakout_1dATRRegime_1dVolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate 1d ATR regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # True Range for ATR
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14) and ATR(50) using Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_smooth_14 = wilder_smooth(tr, 14)
    tr_smooth_50 = wilder_smooth(tr, 50)
    atr_14 = tr_smooth_14
    atr_50 = tr_smooth_50
    
    # Low volatility regime: ATR(14)/ATR(50) > 0.8 (meaning recent volatility not too compressed)
    atr_ratio = atr_14 / atr_50
    low_vol_regime = atr_ratio > 0.8
    
    # 1d volume confirmation filter
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = vol_1d > (1.5 * vol_ma_20_1d)
    
    # Align to 12h timeframe
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_dc, lower_dc = donchian_channel(high, low, 20)
    mid_dc = (upper_dc + lower_dc) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(low_vol_regime_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(upper_dc[i]) or
            np.isnan(lower_dc[i]) or
            np.isnan(mid_dc[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian AND low vol regime AND volume confirmation
            if (low[i] <= upper_dc[i] and close[i] > upper_dc[i] and 
                low_vol_regime_aligned[i] > 0.5 and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian AND low vol regime AND volume confirmation
            elif (high[i] >= lower_dc[i] and close[i] < lower_dc[i] and 
                  low_vol_regime_aligned[i] > 0.5 and 
                  volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to midpoint of Donchian channel
            if close[i] <= mid_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to midpoint of Donchian channel
            if close[i] >= mid_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals