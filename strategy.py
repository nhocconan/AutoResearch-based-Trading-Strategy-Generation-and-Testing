#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_ema_crossover_volume_filter_v1
# Uses 50-day EMA and 200-day EMA crossover on daily chart for trend direction.
# Enters long when 50 EMA crosses above 200 EMA with volume confirmation (volume > 1.5x 20-day average).
# Enters short when 50 EMA crosses below 200 EMA with volume confirmation.
# Uses weekly ADX > 25 to filter for strong trends, avoiding false signals in weak trends or ranges.
# Designed for low trade frequency (target: 10-30 trades/year) to minimize fee drift.
# Works in bull markets (golden cross continuation) and bear markets (death cross continuation).

name = "1d_1w_ema_crossover_volume_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 50 EMA and 200 EMA on daily close
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).values
    
    # Calculate crossover signals
    ema_crossover = np.where(ema_50 > ema_200, 1, -1)  # 1 for golden cross, -1 for death cross
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Weekly ADX calculation (using Wilder's smoothing)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    # Wilder's smoothing function
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    plus_di = np.where(atr_1w != 0, 100 * plus_dm_smooth / atr_1w, 0)
    minus_di = np.where(atr_1w != 0, 100 * minus_dm_smooth / atr_1w, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1w = wilders_smooth(dx, 14)
    
    # Align weekly ADX to daily timeframe (only use completed weekly bars)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    adx_filter = adx_aligned > 25  # strong trend only
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # start after warmup
        # Skip if EMA or ADX not ready
        if np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or np.isnan(adx_filter[i]):
            signals[i] = 0.0
            continue
        
        # Require both volume and strong trend filters
        if not (vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: 50 EMA crosses above 200 EMA (golden cross) with volume
        if ema_crossover[i] == 1 and ema_crossover[i-1] == -1 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: 50 EMA crosses below 200 EMA (death cross) with volume
        elif ema_crossover[i] == -1 and ema_crossover[i-1] == 1 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite crossover
        elif ema_crossover[i] == -1 and ema_crossover[i-1] == 1 and position == 1:
            position = 0
            signals[i] = 0.0
        elif ema_crossover[i] == 1 and ema_crossover[i-1] == -1 and position == -1:
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