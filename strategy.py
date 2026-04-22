#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian channel breakout with daily trend filter and volume confirmation.
Long when price breaks above 4h Donchian upper band (20-period) and daily EMA(34) is rising.
Short when price breaks below 4h Donchian lower band and daily EMA(34) is falling.
Requires volume spike (>1.5x 20-period average) for entry confirmation.
Exits when price returns to Donchian middle band or volatility expands (ATR ratio > 2.0).
Designed for low trade frequency (19-50/year) with strong trend-following edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high)
    low_roll = pd.Series(low)
    donchian_high = high_roll.rolling(window=20, min_periods=20).max().values
    donchian_low = low_roll.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Daily EMA34 for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 40:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    ema34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility exit (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + daily uptrend + volume spike
            if close[i] > donchian_high[i] and ema34_daily_aligned[i] > ema34_daily_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + daily downtrend + volume spike
            elif close[i] < donchian_low[i] and ema34_daily_aligned[i] < ema34_daily_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle band OR volatility expansion (ATR ratio > 2.0)
            exit_signal = False
            
            if position == 1:
                # Exit long: price below middle OR ATR expansion
                if close[i] < donchian_mid[i] or atr[i] > 2.0 * atr[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above middle OR ATR expansion
                if close[i] > donchian_mid[i] or atr[i] > 2.0 * atr[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0