#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d ADX trend filter and volume confirmation
- Long when Williams %R(14) crosses above -80 (oversold bounce) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average
- Short when Williams %R(14) crosses below -20 (overbought rejection) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average
- Exit when Williams %R crosses opposite extreme (-20 for long exit, -80 for short exit) or ADX < 20 (trend weakening)
- Uses 1d ADX for HTF trend alignment to ensure we trade with the higher timeframe trend
- Williams %R extremes provide mean reversion entries within strong trends
- Designed for both bull and bear markets: ADX filter ensures we only trade when trending
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    # ADX calculation requires +DI and -DI
    # +DI = 100 * smoothed +DM / ATR
    # -DI = 100 * smoothed -DM / ATR
    # ADX = smoothed DX where DX = 100 * |+DI - -DI| / (+DI + -DI)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need 30 for ADX, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R signals
        williams_r_cross_above_80 = (williams_r[i-1] <= -80) and (williams_r[i] > -80)  # Oversold bounce
        williams_r_cross_below_20 = (williams_r[i-1] >= -20) and (williams_r[i] < -20)  # Overbought rejection
        
        williams_r_below_80 = williams_r[i] < -80  # Still oversold
        williams_r_above_20 = williams_r[i] > -20  # Still overbought
        
        # Trend filter (using 1d ADX)
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] < 20
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R oversold bounce + strong trend + volume confirmation
            if williams_r_cross_above_80 and strong_trend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought rejection + strong trend + volume confirmation
            elif williams_r_cross_below_20 and strong_trend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -20 (overbought) OR trend weakens
                if williams_r[i] >= -20 or weak_trend:
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R crosses below -80 (oversold) OR trend weakens
                if williams_r[i] <= -80 or weak_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADX_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0