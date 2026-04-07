#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R with 1d trend filter and volume confirmation
# Hypothesis: Williams %R identifies overbought/oversold conditions for mean reversion.
# In trending markets (1d ADX > 25), we fade extremes only when aligned with trend.
# Volume confirms institutional participation. Works in bull/bear via adaptive filtering.
# Target: 12-37 trades/year (50-150 total over 4 years).
name = "6h_williamsr_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend strength
    # True Range
    tr1 = df_1d['high'][1:] - df_1d['low'][1:]
    tr2 = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
    tr3 = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
    tr = np.concatenate([[np.max([df_1d['high'].iloc[0] - df_1d['low'].iloc[0], 
                                  np.abs(df_1d['high'].iloc[0] - df_1d['close'].iloc[0]),
                                  np.abs(df_1d['low'].iloc[0] - df_1d['close'].iloc[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    up_move = df_1d['high'][1:] - df_1d['high'][:-1]
    down_move = df_1d['low'][:-1] - df_1d['low'][1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    tr_smooth = WilderSmooth(tr, 14)
    plus_dm_smooth = WilderSmooth(plus_dm, 14)
    minus_dm_smooth = WilderSmooth(minus_dm, 14)
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = WilderSmooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate daily 20-period volume moving average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Williams %R(14) on 6h data
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close) / (highest_high - lowest_low), 
                          -50)  # Neutral when no range
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Trend filter: only trade in direction of trend when ADX > 25
        # We need close price to determine trend direction from 1d data
        # Approximate: use 1d close trend via EMA crossover
        if i >= len(align_htf_to_ltf(prices, df_1d, df_1d['close'].values)):
            # Fallback: use price action if 1d data not aligned
            trend_up = close[i] > close[i-1]  # Simple proxy
        else:
            # Get aligned 1d close for trend direction
            close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
            trend_up = close_1d_aligned[i] > close_1d_aligned[i-1] if i > 0 else True
        
        if position == 1:  # Long position
            # Exit: Williams %R returns above -20 (overbought) or trend reversal
            if williams_r[i] > -20 or (adx_aligned[i] > 25 and not trend_up):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -80 (oversold) or trend reversal
            if williams_r[i] < -80 or (adx_aligned[i] > 25 and trend_up):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Williams %R < -80 (oversold) + volume confirmation + trend alignment
            # In strong trend (ADX>25), only go long if trend is up
            # In weak trend (ADX<=25), mean reversion from oversold
            if williams_r[i] < -80 and vol_confirm:
                if adx_aligned[i] <= 25 or trend_up:  # Mean reversion or trend-aligned
                    position = 1
                    signals[i] = 0.25
            # Enter short: Williams %R > -20 (overbought) + volume confirmation + trend alignment
            elif williams_r[i] > -20 and vol_confirm:
                if adx_aligned[i] <= 25 or not trend_up:  # Mean reversion or trend-aligned
                    position = -1
                    signals[i] = -0.25
    
    return signals