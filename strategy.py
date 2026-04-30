#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets (ADX < 25),
# we fade extremes: long when %R < -80, short when %R > -20. In trending markets (ADX >= 25),
# we only take trades in direction of 1d EMA34: long if price > EMA34 and %R < -80,
# short if price < EMA34 and %R > -20. Volume spike confirms momentum. Works in bull via
# selective longs, in bear via selective shorts. Discrete sizing 0.25 minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_ME_1dEMA34_ADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d Williams %R(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    
    # Align 1d Williams %R to 6h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1d ADX(14) for regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
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
    
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14, 34)  # warmup for volume MA, Williams %R, EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                if curr_adx < 25:  # Ranging market: fade extremes
                    # Long when oversold
                    if curr_williams_r < -80:
                        signals[i] = 0.25
                        position = 1
                        entry_price = curr_close
                    # Short when overbought
                    elif curr_williams_r > -20:
                        signals[i] = -0.25
                        position = -1
                        entry_price = curr_close
                else:  # Trending market: only trade with EMA trend
                    # Long if price above EMA and oversold
                    if curr_close > curr_ema_34 and curr_williams_r < -80:
                        signals[i] = 0.25
                        position = 1
                        entry_price = curr_close
                    # Short if price below EMA and overbought
                    elif curr_close < curr_ema_34 and curr_williams_r > -20:
                        signals[i] = -0.25
                        position = -1
                        entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price crosses below EMA or Williams %R returns to neutral
            if curr_close < curr_ema_34 or curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price crosses above EMA or Williams %R returns to neutral
            if curr_close > curr_ema_34 or curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals