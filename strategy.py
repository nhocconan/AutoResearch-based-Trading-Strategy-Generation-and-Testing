#!/usr/bin/env python3
"""
1d Donchian breakout + weekly ADX trend + volume confirmation.
Long when price breaks above Donchian(20) high in ADX>20 uptrend (price above 200 EMA).
Short when price breaks below Donchian(20) low in ADX>20 downtrend (price below 200 EMA).
Volume must be above 20-period average to confirm breakouts.
Target: 20-50 total trades over 4 years (5-12/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_adx_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    one_w_close = df_1w['close'].values
    one_w_high = df_1w['high'].values
    one_w_low = df_1w['low'].values
    # ADX calculation
    plus_dm = np.diff(one_w_high, prepend=one_w_high[0])
    minus_dm = np.diff(one_w_low, prepend=one_w_low[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = np.maximum(np.abs(np.diff(one_w_high, prepend=one_w_high[0])),
                    np.maximum(np.abs(np.diff(one_w_low, prepend=one_w_low[0])),
                               np.abs(np.diff(one_w_close, prepend=one_w_close[0]))))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    # EMA200 for trend direction
    ema200 = pd.Series(one_w_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    ema200_aligned = align_htf_to_ltf(prices, df_1w, ema200)
    
    # === 1D DONCHIAN CHANNEL ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        if np.isnan(adx_aligned[i]) or np.isnan(ema200_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly ADX and EMA200
        strong_trend = adx_aligned[i] > 20
        uptrend = strong_trend and close[i] > ema200_aligned[i]
        downtrend = strong_trend and close[i] < ema200_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend weakens
            if close[i] < donchian_low[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend weakens
            if close[i] > donchian_high[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on weekly trend
            if uptrend:
                # In uptrend: long on break above Donchian high
                if high[i] > donchian_high[i] and close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
            elif downtrend:
                # In downtrend: short on break below Donchian low
                if low[i] < donchian_low[i] and close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals