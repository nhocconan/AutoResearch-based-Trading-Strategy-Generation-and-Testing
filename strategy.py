#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume spike confirmation + ATR(14) stoploss.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 2.0 * 4h volume MA(20).
- Exit: ATR-based stoploss (2.5 * ATR(14)) from entry price; no profit target (let trend run).
- Signal size: 0.30 discrete to balance reward and fee drag.
- Designed to capture strong trends in both bull and bear markets via Donchian breakouts filtered by 1d trend and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian(20), volume MA(20), and ATR(14)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian(20) channels for 4h timeframe
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for 4h timeframe
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_4h[0] - low_4h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 4h timeframe
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 needs 34, Donchian needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above Donchian(20) high AND price > 1d EMA34 (uptrend)
                if curr_high > highest_20[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Short: price breaks below Donchian(20) low AND price < 1d EMA34 (downtrend)
                elif curr_low < lowest_20[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check stoploss
            stoploss = entry_price - 2.5 * curr_atr
            if curr_close < stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: check stoploss
            stoploss = entry_price + 2.5 * curr_atr
            if curr_close > stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0