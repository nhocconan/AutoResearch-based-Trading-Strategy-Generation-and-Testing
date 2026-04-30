#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions on 6h chart. In ranging markets (ADX < 25 on 1d),
# we fade extremes: long when %R < -80, short when %R > -20. In trending markets (ADX >= 25),
# we only take trades in direction of 1d EMA34 trend. Volume spike (1.5x 20-period average) confirms
# signal validity. This combines mean reversion in ranges with trend continuation, adapting to
# BTC/ETH market regimes. Discrete sizing 0.25 balances risk and minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_ME_1dEMA34_ADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d EMA34 for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1d ADX(14) for regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where(
        (df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']),
        np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0),
        0
    )
    dm_minus = np.where(
        (df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)),
        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0),
        0
    )
    
    # Smoothed DM
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R on 6h (14-period)
    if len(prices) < 14:
        return np.zeros(n)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_ema = ema_34_aligned[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Ranging market (ADX < 25): mean reversion at extremes
                if curr_adx < 25:
                    if curr_wr < -80:  # Oversold -> long
                        signals[i] = 0.25
                        position = 1
                    elif curr_wr > -20:  # Overbought -> short
                        signals[i] = -0.25
                        position = -1
                # Trending market (ADX >= 25): only trade with EMA trend
                else:
                    if curr_close > curr_ema and curr_wr < -50:  # Uptrend + pullback -> long
                        signals[i] = 0.25
                        position = 1
                    elif curr_close < curr_ema and curr_wr > -50:  # Downtrend + rally -> short
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position
            # Exit when overbought or trend reverses
            if curr_wr > -20 or curr_close < curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when oversold or trend reverses
            if curr_wr < -80 or curr_close > curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals