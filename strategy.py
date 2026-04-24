#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h ADX trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for ADX(14) > 25 to filter strong trends, avoiding choppy markets.
- Entry: Long when price breaks above Donchian(20) high AND 12h ADX > 25 AND volume > 1.5 * 6h volume MA(20);
         Short when price breaks below Donchian(20) low AND 12h ADX > 25 AND volume > 1.5 * 6h volume MA(20).
- Exit: Opposite Donchian breakout (price crosses opposite channel boundary).
- Signal size: 0.25 discrete to control fee drag.
- Designed to capture strong trending moves while avoiding false breakouts in ranging markets.
- Works in bull markets via longs on upward breakouts, bear markets via shorts on downward breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channel calculations
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian Channel on 6h data (20-period)
    period20_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian components to primary 6h timeframe
    upper_6h_aligned = align_htf_to_ltf(prices, df_6h, period20_high)
    lower_6h_aligned = align_htf_to_ltf(prices, df_6h, period20_low)
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h data
    # True Range
    tr1 = pd.Series(high_12h - low_12h).values
    tr2 = np.abs(pd.Series(high_12h).shift(1).values - pd.Series(close_12h).shift(1).values)
    tr3 = np.abs(pd.Series(low_12h).shift(1).values - pd.Series(close_12h).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((pd.Series(high_12h).diff().values > pd.Series(low_12h).diff().values) & 
                       (pd.Series(high_12h).diff().values > 0), 
                       pd.Series(high_12h).diff().values, 0)
    dm_minus = np.where((pd.Series(low_12h).diff().values > pd.Series(high_12h).diff().values) & 
                        (pd.Series(low_12h).diff().values > 0), 
                        pd.Series(low_12h).diff().values, 0)
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period_adx = 14
    atr = wilders_smoothing(tr, period_adx)
    dm_plus_smooth = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smooth = wilders_smoothing(dm_minus, period_adx)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period_adx)
    
    # Align ADX to primary 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate volume MA(20) for 6h timeframe
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (max of 20 for Donchian, 20 for volume, 27 for ADX)
    start_idx = max(20, 20, 27)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_6h_aligned[i]) or np.isnan(lower_6h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Donchian breakout conditions
        upper_breakout = curr_close > upper_6h_aligned[i]
        lower_breakout = curr_close < lower_6h_aligned[i]
        
        # ADX trend filter (strong trend)
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and strong_trend:
                # Long: Price breaks above upper Donchian band
                if upper_breakout:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below lower Donchian band
                elif lower_breakout:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below lower Donchian band (opposite breakout)
            if lower_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above upper Donchian band (opposite breakout)
            if upper_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0