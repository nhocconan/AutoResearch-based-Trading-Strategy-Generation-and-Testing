#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h volume spike and 1d ADX trend filter
# Camarilla pivot levels (R1/S1) provide tight support/resistance for precise entries
# Breakout above R1 or below S1 with volume confirmation (2x 20-bar EMA) captures momentum
# 1d ADX > 20 ensures alignment with higher timeframe trend to avoid range-bound whipsaws
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Designed for 1h timeframe targeting 15-37 trades/year (60-150 total over 4 years)
# Uses discrete position sizing (0.20) to minimize fee churn and control drawdown
# Works in bull markets (breakout above R1 + 1d ADX up-trend) and bear markets (breakout below S1 + 1d ADX down-trend)

name = "1h_Camarilla_R1S1_Breakout_4hVolume_1dADX_Trend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC) - prices.index is already DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 4h volume EMA for confirmation
    vol_ema_20_4h = pd.Series(df_4h['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation_4h = df_4h['volume'].values > (2.0 * vol_ema_20_4h)
    volume_confirmation_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_confirmation_4h)
    
    # 1d data for ADX trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 1d ADX calculation (using standard Wilder's smoothing)
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['high'])).abs()
    tr4 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
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
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate Camarilla levels for each 1d bar (based on same day's OHLC)
    # Standard Camarilla: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_r1 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12
    camarilla_s1 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (use same day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_confirmation_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX (trending market filter)
        trending_market = adx_1d_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R1 with volume confirmation and trending market
            if close[i] > camarilla_r1_aligned[i] and trending_market and volume_confirmation_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Breakout below S1 with volume confirmation and trending market
            elif close[i] < camarilla_s1_aligned[i] and trending_market and volume_confirmation_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S1 (reversal) OR market loses trend
            if close[i] < camarilla_s1_aligned[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R1 (reversal) OR market loses trend
            if close[i] > camarilla_r1_aligned[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals