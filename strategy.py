#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout + Volume Confirmation + ADX Trend Filter
# Hypothesis: Price breaks out of 20-period Donchian channel with volume confirmation (volume > 1.5x average) and ADX > 20 (trending market).
# Works in bull/bear by capturing breakouts with trend confirmation. Target: 50-150 total trades over 4 years (12-37/year).
# Uses 1d trend filter via EMA(50) to avoid counter-trend trades.

name = "12h_donchian_breakout_vol_adx_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Donchian Channel (20-period high/low)
    donch_period = 20
    highest_high = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lowest_low = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # ADX (14-period) for trend strength
    adx_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=adx_period, min_periods=adx_period).mean().values
    
    up_move = pd.Series(high).diff()
    down_move = pd.Series(low).diff() * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / (atr * adx_period)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / (atr * adx_period)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or trend weakens
            if close[i] < lowest_low[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or trend weakens
            if close[i] > highest_high[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume and trend filters
            vol_ok = vol_ratio[i] > 1.5
            adx_ok = adx[i] > 20
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Breakout above upper band with volume and trend
            if close[i] > highest_high[i] and vol_ok and adx_ok and uptrend:
                position = 1
                signals[i] = 0.25
            # Breakdown below lower band with volume and trend
            elif close[i] < lowest_low[i] and vol_ok and adx_ok and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals