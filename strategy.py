#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R extreme reversal with volume confirmation and ADX regime filter
# Long when 1d Williams %R < -80 (oversold) AND price > 12h EMA(50) AND volume > 1.5 * avg_volume(20) AND ADX(14) < 25 (range/low trend)
# Short when 1d Williams %R > -20 (overbought) AND price < 12h EMA(50) AND volume > 1.5 * avg_volume(20) AND ADX(14) < 25
# Exit when Williams %R returns to -50 (mean reversion) or ADX > 30 (strong trend)
# Uses discrete sizing 0.25 to balance return and drawdown
# Williams %R identifies exhaustion points; volume confirms participation; ADX avoids strong trends where mean reversion fails
# Works in bull (buy dips in range) and bear (sell rallies in range) markets
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_1dWilliamsR_Extreme_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed daily bars for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    
    # Align 1d Williams %R to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 12h EMA(50) for trend filter
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate ADX(14) for regime filter (avoid strong trends)
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - low[:-1])
    tr3 = np.abs(low[1:] - high[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Pad arrays to match length
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    tr = np.concatenate([[np.nan], tr])
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev_smoothed * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            else:
                result[i] = result[i-1]
        return result
    
    period = 14
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    tr_smooth = wilders_smoothing(tr, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above EMA50 + volume spike + low ADX (<25)
            if (williams_r_aligned[i] < -80 and close[i] > ema_50[i] and 
                volume_confirm[i] and adx[i] < 25):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + price below EMA50 + volume spike + low ADX (<25)
            elif (williams_r_aligned[i] > -20 and close[i] < ema_50[i] and 
                  volume_confirm[i] and adx[i] < 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) OR ADX > 30 (strong trend)
            if williams_r_aligned[i] >= -50 or adx[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) OR ADX > 30 (strong trend)
            if williams_r_aligned[i] <= -50 or adx[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals