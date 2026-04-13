#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for primary trend direction (use once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    vol_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channel (20-period)
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR (14-period) for volatility filter
    tr1 = np.abs(high_4h[1:] - low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 4h indicators to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, high_max_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, low_min_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    # Get 1d data for regime filter (chop/ADX-like)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period) for trend strength
    # True Range
    tr1_1d = np.abs(high_1d[1:] - low_1d[:-1])
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI values
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1h momentum for entry timing
    mom_10 = pd.Series(close).diff(10).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    # Pre-compute session filter (8-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(mom_10[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (position_size if position == 1 else -position_size)
            continue
        
        # Trend direction from 4h Donchian breakout
        bullish_breakout = close[i] > donchian_upper_aligned[i]
        bearish_breakout = close[i] < donchian_lower_aligned[i]
        
        # Volatility filter: only trade when volatility is elevated
        vol_filter = atr_14_aligned[i] > np.nanmedian(atr_14_aligned[max(0, i-100):i])
        
        # Regime filter: only trade in trending markets (ADX > 25)
        regime_filter = adx_aligned[i] > 25
        
        # Entry timing with 1h momentum
        mom_long = mom_10[i] > 0
        mom_short = mom_10[i] < 0
        
        # Entry conditions
        long_entry = bullish_breakout and vol_filter and regime_filter and mom_long
        short_entry = bearish_breakout and vol_filter and regime_filter and mom_short
        
        # Exit conditions: opposite breakout or momentum reversal
        exit_long = position == 1 and (bearish_breakout or mom_10[i] < 0)
        exit_short = position == -1 and (bullish_breakout or mom_10[i] > 0)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_donchian_breakout_adx_mom"
timeframe = "1h"
leverage = 1.0