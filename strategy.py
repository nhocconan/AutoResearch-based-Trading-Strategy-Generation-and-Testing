#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ADX trend filter and volume confirmation
# Donchian(20) breakout captures strong momentum moves in both bull and bear markets
# 1w ADX > 25 ensures alignment with weekly trend to avoid range-bound whipsaws
# Volume confirmation (2x 20-bar EMA) filters false breakouts
# Designed for 1d timeframe targeting 7-25 trades/year (30-100 total over 4 years)
# Uses discrete position sizing (0.30) to minimize fee churn and control drawdown
# Works in bull markets (breakout above upper band + 1w ADX up-trend) and bear markets (breakout below lower band + 1w ADX down-trend)

name = "1d_Donchian20_1wADX_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 1w ADX calculation (using standard Wilder's smoothing)
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = pd.Series(df_1w['low']).diff().abs()
    tr3 = (pd.Series(df_1w['close']).shift() - pd.Series(df_1w['high'])).abs()
    tr4 = (pd.Series(df_1w['close']).shift() - pd.Series(df_1w['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1w['high']).diff()
    down_move = -pd.Series(df_1w['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    # 1d Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w ADX (trending market filter)
        trending_market = adx_1w_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian band with volume confirmation and trending market
            if close[i] > highest_high[i] and trending_market and volume_confirmation[i]:
                signals[i] = 0.30
                position = 1
            # Short: Breakout below lower Donchian band with volume confirmation and trending market
            elif close[i] < lowest_low[i] and trending_market and volume_confirmation[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian band (reversal) OR market loses trend
            if close[i] < lowest_low[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian band (reversal) OR market loses trend
            if close[i] > highest_high[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals