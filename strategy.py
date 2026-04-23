#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1h ADX trend filter.
Long when price breaks above 4h Donchian upper (20) AND volume > 1.5x 20-period average AND 1h ADX > 25.
Short when price breaks below 4h Donchian lower (20) AND volume > 1.5x 20-period average AND 1h ADX > 25.
Exit when price retraces to 4h Donchian middle OR ATR stoploss (2.0*ATR).
Uses 08-20 UTC session filter and discrete position sizing (0.20) to minimize fee churn.
Designed for 1h timeframe targeting 15-37 trades/year per symbol (60-150 total over 4 years).
Works in both bull and bear markets by requiring ADX > 25 to ensure trending conditions and volume confirmation to filter false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Calculate 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels: upper = max(high, 20), lower = min(low, 20), middle = (upper+lower)/2
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_4h = (upper_4h + lower_4h) / 2.0
    
    # Align Donchian levels to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_4h_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # Calculate 1h ADX (14) for trend filter
    # ADX calculation requires +DI, -DI, and DX
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation (re-use from ADX)
    atr_val = atr  # already calculated above
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 14, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(middle_4h_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_val[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr_val[i]
        upper = upper_4h_aligned[i]
        lower = lower_4h_aligned[i]
        middle = middle_4h_aligned[i]
        adx_val = adx[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian upper AND volume spike AND ADX > 25 (trending)
            if (price > upper and 
                volume[i] > 1.5 * vol_ma_val and 
                adx_val > 25):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: Price breaks below 4h Donchian lower AND volume spike AND ADX > 25 (trending)
            elif (price < lower and 
                  volume[i] > 1.5 * vol_ma_val and 
                  adx_val > 25):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 4h Donchian middle
            if position == 1 and price <= middle:
                exit_signal = True
            elif position == -1 and price >= middle:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_DonchianBreakout_Volume_ADX25_TrendFilter_ATRStop"
timeframe = "1h"
leverage = 1.0