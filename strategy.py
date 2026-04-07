#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    daily_ema = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or np.isnan(daily_ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian or trend reversal
            if close[i] < lower[i] or not above_daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian or trend reversal
            if close[i] > upper[i] or not below_daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long entry: break above upper Donchian + volume + daily uptrend
            if close[i] > upper[i] and vol_confirm and above_daily_ema:
                position = 1
                signals[i] = 0.30
            # Short entry: break below lower Donchian + volume + daily downtrend
            elif close[i] < lower[i] and vol_confirm and below_daily_ema:
                position = -1
                signals[i] = -0.30
    
    return signals