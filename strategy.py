#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Donchian breakout with weekly ADX trend filter and volume confirmation
# Uses Donchian(20) channel breakout for entry, weekly ADX(14) for trend strength,
# and volume > 1.5x 20-day average for confirmation. Exits on opposite Donchian break.
# Designed for low trade frequency (< 25/year) to minimize fee drag.
# Works in trending markets via breakouts and avoids choppy periods via ADX filter.

name = "donchian20_weekly_adx_volume_v1"
timeframe = "1d"
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
    
    # Weekly data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX calculation on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) != 0, dx, 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend strength filter: ADX > 25
        strong_trend = adx_1w_aligned[i] > 25
        
        # Volume confirmation: volume > 1.5x 20-day average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Long breakout: close above upper Donchian band
        long_breakout = close[i] > highest_high[i]
        
        # Short breakout: close below lower Donchian band
        short_breakout = close[i] < lowest_low[i]
        
        # Entry conditions
        if long_breakout and strong_trend and vol_confirmed:
            signals[i] = 0.30
        elif short_breakout and strong_trend and vol_confirmed:
            signals[i] = -0.30
        # Exit on opposite Donchian break
        elif (signals[i-1] > 0 and short_breakout) or (signals[i-1] < 0 and long_breakout):
            signals[i] = 0.0
        else:
            signals[i] = signals[i-1]  # hold position
    
    return signals