#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX25 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, 1d ADX > 25, and volume > 1.5x 20-bar avg.
# Short when price breaks below Donchian(20) low, 1d ADX > 25, and volume > 1.5x 20-bar avg.
# Exit when price crosses the 20-period EMA (trend exhaustion).
# Uses 1d ADX for higher timeframe trend strength, targeting 12-37 trades/year on 6h.
# Trend filter ensures we only trade in strong trends, volume confirmation reduces false signals.
# Works in bull markets via breakouts and in bear markets via breakdowns with trend alignment.

name = "6h_Donchian20_1dADX25_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed / period) + current
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, (dm_plus_smooth / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d != 0, (dm_minus_smooth / atr_1d) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, period)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian(20) channels on 6h data
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # 20-period EMA for exit signal
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_20[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_adx_1d = adx_1d_aligned[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        curr_ema_20 = ema_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = curr_adx_1d > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper, strong trend, volume spike
            if (curr_close > curr_donchian_upper and 
                strong_trend and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, strong trend, volume spike
            elif (curr_close < curr_donchian_lower and 
                  strong_trend and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below 20-period EMA (trend exhaustion)
            if curr_close < curr_ema_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above 20-period EMA (trend exhaustion)
            if curr_close > curr_ema_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals