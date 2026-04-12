#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_adaptive_kelly_breakout_v1
# Uses weekly price channels (Donchian) for trend direction and daily ATR for volatility-based position sizing.
# Enters on 6h breakouts of weekly Donchian channels with volume confirmation.
# Position size adapts to volatility (inverse ATR) and trend strength (ADX) using Kelly-inspired scaling.
# Designed for low trade frequency (12-37/year) to minimize fee drift while adapting to market conditions.
# Works in bull markets (trend continuation) and bear markets (trend reversals at extremes).

name = "6h_1w_1d_adaptive_kelly_breakout_v1"
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
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for ATR and trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_20 = df_1w['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align weekly channels to 6h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Daily ATR (14-period) for volatility normalization
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily ADX (14-period) for trend strength
    plus_dm = np.where((df_1d['high'].values[1:] - df_1d['high'].values[:-1]) > 
                       (df_1d['low'].values[:-1] - df_1d['low'].values[1:]), 
                       np.maximum(df_1d['high'].values[1:] - df_1d['high'].values[:-1], 0), 0)
    minus_dm = np.where((df_1d['low'].values[:-1] - df_1d['low'].values[1:]) > 
                        (df_1d['high'].values[1:] - df_1d['high'].values[:-1]), 
                        np.maximum(df_1d['low'].values[:-1] - df_1d['low'].values[1:], 0), 0)
    
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_dx = wilders_smooth(tr, 14)
    plus_di_smooth = wilders_smooth(plus_dm, 14)
    minus_di_smooth = wilders_smooth(minus_dm, 14)
    
    plus_di = np.where(atr_dx != 0, 100 * plus_di_smooth / atr_dx, 0)
    minus_di = np.where(atr_dx != 0, 100 * minus_di_smooth / atr_dx, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = wilders_smooth(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5 * 50-period average (6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate adaptive position size based on volatility and trend strength
        # Inverse volatility scaling (lower vol = larger position)
        vol_scaling = np.clip(1.0 / (atr_1d_aligned[i] * 0.01), 0.5, 2.0)
        # Trend strength scaling (stronger trend = larger position)
        trend_scaling = np.clip(adx_1d_aligned[i] / 25.0, 0.5, 1.5)
        # Base size scaled by both factors
        base_size = 0.25
        adaptive_size = base_size * vol_scaling * trend_scaling
        # Clamp to reasonable range
        adaptive_size = np.clip(adaptive_size, 0.15, 0.35)
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = adaptive_size
            elif position == -1:
                signals[i] = -adaptive_size
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly Donchian high
        if close[i] > donchian_high[i] and position != 1:
            position = 1
            signals[i] = adaptive_size
        # Short signal: price breaks below weekly Donchian low
        elif close[i] < donchian_low[i] and position != -1:
            position = -1
            signals[i] = -adaptive_size
        # Exit conditions: opposite breakout or volatility expansion
        elif close[i] < donchian_low[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > donchian_high[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = adaptive_size
            elif position == -1:
                signals[i] = -adaptive_size
            else:
                signals[i] = 0.0
    
    return signals