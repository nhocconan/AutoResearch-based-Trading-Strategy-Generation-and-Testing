#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike_ATRStop_v2
Hypothesis: Daily Donchian(20) breakouts filtered by weekly EMA20 trend and volume spike (>2.0x 20-period average).
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to minimize fee churn.
Designed for 10-20 trades/year per symbol, targeting BTC/ETH robustness in bull/bear regimes.
Weekly trend filter reduces whipsaws during sideways markets while capturing strong directional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for EMA20 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Daily OHLC for Donchian calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_channel = high_roll.values
    lower_channel = low_roll.values
    
    # === Weekly EMA20 for trend filter ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) 
            or np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume filter: current volume > 2.0x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_filter = volume[i] > 2.0 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > Upper Donchian, weekly uptrend, volume filter
            long_breakout = price > upper_channel[i]
            long_trend = price > ema_20_1w_aligned[i]
            
            # Short conditions: price < Lower Donchian, weekly downtrend, volume filter
            short_breakout = price < lower_channel[i]
            short_trend = price < ema_20_1w_aligned[i]
            
            # Entry logic - ONLY enter on volume filter + trend alignment
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below Lower Donchian (breakdown)
            elif price < lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above Upper Donchian (breakout)
            elif price > upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike_ATRStop_v2"
timeframe = "1d"
leverage = 1.0