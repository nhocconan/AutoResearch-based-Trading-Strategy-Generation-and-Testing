#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme + 1d EMA34 Trend + Volume Spike + Session Filter.
# Williams %R < -80 = oversold (long), > -20 = overbought (short) on 4h.
# 1d EMA34 determines trend: price > EMA34 = uptrend, price < EMA34 = downtrend.
# Volume spike: current 4h volume > 2.0x 20-period volume MA.
# Session: 08-20 UTC (avoid low-volume Asian session).
# Only take longs in uptrend when oversold, shorts in downtrend when overbought.
# Exit on opposite Williams %R extreme or trend change.
# Designed for mean reversion in strong trends with low trade frequency.

name = "4h_WilliamsR_Extreme_1dEMA34_VolumeSpike_Session"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current 4h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 2.0)
        
        # 1d EMA34 trend conditions
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Williams %R extremes
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        if position == 0:
            # Long: oversold AND uptrend AND volume spike AND session
            if oversold and price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: overbought AND downtrend AND volume spike AND session
            elif overbought and price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: overbought OR price crosses below EMA34
            if overbought or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: oversold OR price crosses above EMA34
            if oversold or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals