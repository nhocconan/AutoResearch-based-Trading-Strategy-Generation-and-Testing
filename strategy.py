#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h price action with 4h ADX trend filter and 1d volume regime filter
# Uses ADX > 25 to identify trending markets on 4h, enters on 1h pullbacks to EMA21
# Volume filter requires 1d volume above 20-day average to ensure participation
# Designed for 15-35 trades/year with proper risk control via trend failure
name = "1h_ADX25_Trend_EMA21_Pullback_VolumeRegime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            else:
                result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day average volume on 1d
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 1h EMA21 for pullback entries
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(ema21[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume regime: current 1d volume must be above 20-day average
        # Need to get current 1d volume - use the aligned 1d data
        df_1d_current = get_htf_data(prices, '1d')
        if len(df_1d_current) == 0:
            vol_1d_current = 0
        else:
            # Find the most recent 1d bar that's closed
            idx_1d = len(df_1d_current) - 1
            while idx_1d >= 0 and df_1d_current.iloc[idx_1d]['open_time'] > prices.iloc[i]['open_time']:
                idx_1d -= 1
            if idx_1d < 0:
                vol_1d_current = 0
            else:
                vol_1d_current = df_1d_current.iloc[idx_1d]['volume']
        
        vol_regime = vol_1d_current > vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else False
        
        if position == 0:
            # Look for trend alignment (ADX > 25) and pullback to EMA21
            if adx_aligned[i] > 25 and vol_regime:
                # Long: price pulls back to EMA21 from above
                if close[i] <= ema21[i] * 1.005 and close[i] >= ema21[i] * 0.995:
                    # Check if we're in an uptrend (price above EMA21 recently)
                    if close[i-5] > ema21[i-5] if i >= 5 else False:
                        signals[i] = 0.25
                        position = 1
                # Short: price pulls back to EMA21 from below
                elif close[i] >= ema21[i] * 0.995 and close[i] <= ema21[i] * 1.005:
                    # Check if we're in a downtrend (price below EMA21 recently)
                    if close[i-5] < ema21[i-5] if i >= 5 else False:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: trend fails (ADX < 20) or price breaks EMA21 significantly
            if adx_aligned[i] < 20 or close[i] < ema21[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend fails (ADX < 20) or price breaks EMA21 significantly
            if adx_aligned[i] < 20 or close[i] > ema21[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals