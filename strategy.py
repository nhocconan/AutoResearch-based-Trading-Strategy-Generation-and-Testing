#/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend Filter, Volume Confirmation, and ATR Stop.
Long when price breaks above Donchian upper band and 1d EMA trend is up with volume confirmation.
Short when price breaks below Donchian lower band and 1d EMA trend is down with volume confirmation.
Exit when price crosses back to Donchian middle line.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === ATR (14) for position sizing and volatility filter ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1d EMA trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below middle line
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above middle line
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if vol_ratio[i] < 1.3:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with 1d trend filter and volume
            if close[i] > donchian_high[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                # Breakout above upper band with rising 1d trend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                # Breakdown below lower band with falling 1d trend -> short
                position = -1
                signals[i] = -0.25
    
    return signals