#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d ADX regime filter
# Elder Ray measures bullish/bearish power via EMA(13): Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d ADX > 25 indicates trending regime (use Elder Ray for trend continuation)
# 1d ADX < 20 indicates ranging regime (fade extreme Elder Ray values)
# Volume confirmation (>1.5x 20-period average) filters low-conviction moves
# Target: 15-25 trades/year per symbol with clear regime-based logic
name = "6h_ElderRay_1dADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d ADX for regime detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    plus_dm = high_1d[1:] - high_1d[:-1]
    minus_dm = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[period-1:2*period-1]) if 2*period-1 <= len(data) else np.nan
            # Wilder smoothing: today = (yesterday * (period-1) + today) / period
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    tr_smooth = wilder_smooth(tr, period_adx)
    plus_dm_smooth = wilder_smooth(plus_dm, period_adx)
    minus_dm_smooth = wilder_smooth(minus_dm, period_adx)
    
    # DI+ and DI-
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smooth(dx, period_adx)
    
    adx_1d = adx  # already aligned to 1d index
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Elder Ray components on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Higher = stronger bullish pressure
    bear_power = low - ema_13   # Lower (more negative) = stronger bearish pressure
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1d_aligned[i]
        
        if position == 0:
            # Trending regime (ADX > 25): Elder Ray continuation
            if adx_val > 25:
                # Long: strong bullish power + volume confirmation
                if bull_power[i] > 0 and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: strong bearish power + volume confirmation
                elif bear_power[i] < 0 and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime (ADX < 20): fade extreme Elder Ray values
            elif adx_val < 20:
                # Long: oversold bear power (extreme negative) + volume
                if bear_power[i] < np.percentile(bear_power[max(0, i-50):i+1], 5) and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: overbought bull power (extreme positive) + volume
                elif bull_power[i] > np.percentile(bull_power[max(0, i-50):i+1], 95) and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                    
        elif position == 1:
            # Long exit: weakening bullish power or ADX drops below 20 (range)
            if bull_power[i] <= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: weakening bearish power or ADX drops below 20 (range)
            if bear_power[i] >= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals