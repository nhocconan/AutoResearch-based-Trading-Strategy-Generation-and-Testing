#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + ADX regime filter with 1d HTF confirmation
    # Bull power = High - EMA(13), Bear power = EMA(13) - Low
    # Long when Bull power > 0 and Bear power < 0 and ADX > 25 (trending)
    # Short when Bear power > 0 and Bull power < 0 and ADX > 25 (trending)
    # Exit when power signals reverse or ADX < 20 (range)
    # Uses 1d EMA for smoother trend filter to avoid whipsaws
    # Works in bull (buy strength) and bear (sell weakness) with regime filter
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(13) for 6h (Elder Ray)
    close_6h_series = pd.Series(close_6h)
    ema13 = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_6h - ema13  # Bull power: High - EMA
    bear_power = ema13 - low_6h   # Bear power: EMA - Low
    
    # Calculate ADX(14) for 6h
    # True Range
    tr1 = np.abs(high_6h[1:] - low_6h[1:])
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    
    # Directional Movement
    dm_plus = np.where((high_6h[1:] - high_6h[:-1]) > (low_6h[:-1] - low_6h[1:]), 
                       np.maximum(high_6h[1:] - high_6h[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_6h[:-1] - low_6h[1:]) > (high_6h[1:] - high_6h[:-1]), 
                        np.maximum(low_6h[:-1] - low_6h[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period]) if np.any(~np.isnan(data[1:period])) else 0
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
                else:
                    result[i] = np.nan
        return result
    
    atr = wilder_smoothing(tr, 14)
    dm_plus_smooth = wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr != 0, (dm_minus_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get 1d EMA(20) for HTF trend filter
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Price relative to 1d EMA for trend filter
    price_above_1d_ema = close > ema20_1d_aligned
    price_below_1d_ema = close < ema20_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx[i]) or
            np.isnan(ema20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Elder Ray + ADX + 1d EMA filter
        long_entry = (bull_power[i] > 0 and bear_power[i] < 0 and 
                     adx[i] > 25 and price_above_1d_ema[i] and position != 1)
        short_entry = (bear_power[i] > 0 and bull_power[i] < 0 and 
                      adx[i] > 25 and price_below_1d_ema[i] and position != -1)
        
        # Exit conditions: reverse signals or ADX weakens
        exit_long = (position == 1 and 
                    (bull_power[i] <= 0 or bear_power[i] >= 0 or adx[i] < 20))
        exit_short = (position == -1 and 
                     (bear_power[i] <= 0 or bull_power[i] >= 0 or adx[i] < 20))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0