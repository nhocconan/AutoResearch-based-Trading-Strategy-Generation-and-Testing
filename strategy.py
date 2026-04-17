#!/usr/bin/env python3
"""
1h strategy with 4h and 1d filters for trend and volatility regime.
Trend: 4h EMA21 > EMA50 (bull) or < (bear).
Volatility regime: 1d ATR(14) > SMA(50) of ATR = high vol (breakout mode).
Entry: On 1h, price breaks 4h Donchian(20) high/low in trend direction with volume > 1.5x 20-bar avg.
Exit: Opposite Donchian break or time-based (48 bars).
Only trade 08-20 UTC to avoid low-liquidity hours.
Position size: 0.20.
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
    
    # Get 4h data for trend and Donchian
    df_4h = get_htf_data(prices, '4h')
    # 4h EMA21 and EMA50 for trend
    ema21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    # 4h Donchian(20) for breakout levels
    donch_high_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for volatility regime
    df_1d = get_htf_data(prices, '1d')
    # 1d ATR(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_ma50_1d = pd.Series(atr14_1d).rolling(window=50, min_periods=50).mean().values
    high_vol = atr14_1d > atr_ma50_1d  # high volatility regime
    
    # Align all 4h and 1d factors to 1h
    ema21_4h_1h = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_1h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    donch_high_4h_1h = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_1h = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    high_vol_1h = align_htf_to_ltf(prices, df_1d, high_vol)
    
    # 1h volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    bars_since_entry = 0
    
    start_idx = max(100, 50)  # warmup
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        if np.any([np.isnan(ema21_4h_1h[i]), np.isnan(ema50_4h_1h[i]),
                   np.isnan(donch_high_4h_1h[i]), np.isnan(donch_low_4h_1h[i]),
                   np.isnan(high_vol_1h[i]), np.isnan(vol_ma_20.iloc[i])]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20.iloc[i]
        trend_up = ema21_4h_1h[i] > ema50_4h_1h[i]
        trend_down = ema21_4h_1h[i] < ema50_4h_1h[i]
        vol_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            bars_since_entry = 0
            # Long: uptrend, high vol, break above 4h Donchian high
            if trend_up and high_vol_1h[i] and price > donch_high_4h_1h[i] and vol_ok:
                signals[i] = 0.20
                position = 1
            # Short: downtrend, high vol, break below 4h Donchian low
            elif trend_down and high_vol_1h[i] and price < donch_low_4h_1h[i] and vol_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            bars_since_entry += 1
            # Exit: downtrend, break below 4h Donchian low, or timeout (48 bars)
            if (not trend_up and price < donch_low_4h_1h[i]) or bars_since_entry >= 48:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            bars_since_entry += 1
            # Exit: uptrend, break above 4h Donchian high, or timeout (48 bars)
            if (not trend_down and price > donch_high_4h_1h[i]) or bars_since_entry >= 48:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hTrend_1dVol_Breakout"
timeframe = "1h"
leverage = 1.0