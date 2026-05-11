#!/usr/bin/env python3
name = "6h_Adaptive_Regime_Donchian_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # ATR (14-period) for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # Volume spike filter (volume > 1.5x 20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    # ADX (14-period) for regime detection
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    tr_series = pd.Series(tr)
    atr_14 = tr_series.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm_series.rolling(window=14, min_periods=14).mean() / atr_14)
    minus_di = 100 * (minus_dm_series.rolling(window=14, min_periods=14).mean() / atr_14)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1-day trend filter (EMA 34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    daily_uptrend = close > ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14, 34)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(atr[i]) or np.isnan(volume_ma20[i]) or \
           np.isnan(adx[i]) or np.isnan(daily_uptrend[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Donchian breakout up + ADX > 25 (trending) + daily uptrend + volume
            if close[i] > donchian_upper[i] and adx[i] > 25 and daily_uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + ADX > 25 (trending) + daily downtrend + volume
            elif close[i] < donchian_lower[i] and adx[i] > 25 and not daily_uptrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Donchian breakdown or ADX < 20 (range) or daily trend reversal
            if close[i] < donchian_lower[i] or adx[i] < 20 or not daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Donchian breakout up or ADX < 20 (range) or daily trend reversal
            if close[i] > donchian_upper[i] or adx[i] < 20 or daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals