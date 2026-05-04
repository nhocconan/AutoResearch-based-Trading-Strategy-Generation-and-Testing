#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d ADX trend filter
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20 EMA AND 1d ADX > 25
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20 EMA AND 1d ADX > 25
# Exit when price crosses the Donchian midpoint (10-period average of high/low)
# Uses discrete sizing (0.30) to limit fee churn. Designed for 20-50 trades/year on 4h.
# Works in bull markets via longs on breakouts and bear markets via shorts on breakdowns.
# Volume confirmation avoids false breakouts, ADX filter ensures trending conditions.

name = "4h_Donchian20_Volume_ADX_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF ADX filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_trending = adx > 25  # Strong trend filter
    
    # Align 1d ADX trend to 4h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    
    # Calculate Donchian channels (20-period)
    # Donchian high: highest high over 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low channels
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx_trending_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND 1d trending
            if (close[i] > donchian_high[i] and 
                volume_spike[i] and 
                adx_trending_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND 1d trending
            elif (close[i] < donchian_low[i] and 
                  volume_spike[i] and 
                  adx_trending_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals