#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x 20 EMA
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x 20 EMA
# Williams %R catches mean reversion in extended moves, ADX ensures we only trade in trending markets
# where mean reversion is more likely to work (pullbacks in trends). Volume confirms participation.
# Works in bull markets via longs on pullbacks and bear markets via shorts on rallies.
# Uses discrete sizing (0.25) to limit fee drag. Target: 50-150 total trades over 4 years.

name = "6h_WilliamsR_Extreme_1dADX_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Williams %R and ADX
        return np.zeros(n)
    
    # Get daily OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HHV - Close) / (HHV - LLV)
    period_wr = 14
    hh_v = pd.Series(high_1d).rolling(window=period_wr, min_periods=period_wr).max().values
    ll_v = pd.Series(low_1d).rolling(window=period_wr, min_periods=period_wr).min().values
    williams_r = -100 * (hh_v - close_1d) / (hh_v - ll_v + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Calculate ADX on 1d
    # ADX requires +DI and -DI calculation
    period_adx = 14
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = (pd.Series(close_1d) - pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    
    # +DM and -DM
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    tr_period = pd.Series(tr).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / (tr_period + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_period + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    
    # Williams %R extremes: < -80 oversold, > -20 overbought
    wr_oversold = williams_r < -80
    wr_overbought = williams_r > -20
    
    # ADX > 25 indicates strong trend
    strong_trend = adx > 25
    
    # Align 1d indicators to 6h timeframe
    wr_oversold_aligned = align_htf_to_ltf(prices, df_1d, wr_oversold.astype(float))
    wr_overbought_aligned = align_htf_to_ltf(prices, df_1d, wr_overbought.astype(float))
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(wr_oversold_aligned[i]) or np.isnan(wr_overbought_aligned[i]) or 
            np.isnan(strong_trend_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold AND strong trend AND volume spike
            if (wr_oversold_aligned[i] > 0.5 and 
                strong_trend_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought AND strong trend AND volume spike
            elif (wr_overbought_aligned[i] > 0.5 and 
                  strong_trend_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 (exit oversold) OR trend weakens
            if (wr_oversold_aligned[i] < 0.5 or  # Williams %R no longer oversold
                strong_trend_aligned[i] < 0.5):   # Trend weakened
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 (exit overbought) OR trend weakens
            if (wr_overbought_aligned[i] < 0.5 or  # Williams %R no longer overbought
                strong_trend_aligned[i] < 0.5):    # Trend weakened
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals