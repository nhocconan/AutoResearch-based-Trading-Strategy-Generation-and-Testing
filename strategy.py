#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX) and volume confirmation
    # Long: Bull Power > 0 AND ADX(14) > 20 (trending) AND volume > 1.5x avg
    # Short: Bear Power < 0 AND ADX(14) > 20 (trending) AND volume > 1.5x avg
    # Exit: Opposite power signal OR ADX < 15 (range) OR volume < avg
    # Using 6h timeframe for balanced trade frequency (target 12-37/year), Elder Ray for momentum,
    # ADX to filter weak trends, and volume confirmation to avoid false signals.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    ema13 = None  # Will compute below
    
    # Get daily data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align daily ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 13-period EMA for Elder Ray (primary timeframe)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Get 6h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema13[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 20 indicates trending market (avoid choppy markets)
        trending_market = adx_1d_aligned[i] > 20
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0
        bear_signal = bear_power[i] < 0
        
        # Exit conditions: Opposite signal OR weak trend OR low volume
        exit_long = bear_signal or (adx_1d_aligned[i] < 15) or not volume_spike[i]
        exit_short = bull_signal or (adx_1d_aligned[i] < 15) or not volume_spike[i]
        
        # Entry logic: Elder Ray signal + trending market + volume confirmation
        long_entry = bull_signal and trending_market and volume_spike[i]
        short_entry = bear_signal and trending_market and volume_spike[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0